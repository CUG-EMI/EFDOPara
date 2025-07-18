import os
import argparse  
import numpy as np
import torch
from torchinfo import summary
import torch.distributed as dist
import yaml
from timeit import default_timer
from utilities import *
from EFDO import EFDO
import warnings  
 
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")  

torch.manual_seed(0)
np.random.seed(0)

def cleanup():
    dist.destroy_process_group()

def get_batch_data(TRAIN_PATH, VAL_PATH, TEST_PATH,   
                   ntrain, nval, ntest,   
                   r_train, s_train, r_val, s_val, r_test, s_test,   
                   batch_size, n_out):  
    '''  
    preprocess data for training, validation and testing dataset  

    Parameters:  
    ----------  
        - TRAIN_PATH : path of the training dataset  
        - VAL_PATH   : path of the validation dataset  
        - TEST_PATH  : path of the testing dataset  
        - ntrain     : number of training data   
        - nval       : number of validation data  
        - ntest      : number of testing data   
        - r_train    : downsampling factor of training data  
        - s_train    : resolution of training data  
        - r_val      : downsampling factor of validation data  
        - s_val      : resolution of validation data  
        - r_test     : downsampling factor of testing data  
        - s_test     : resolution of testing data  
        - batch_size : batch size in training and testing  
        - n_out      : number of output channels  
    '''  
    key_map0 = ['rhoxy','phsxy','rhoyx','phsyx']
    if n_out <= 2:
        key_map0 = ['rhoxy','rhoyx']  
    key_map = key_map0[:n_out]  

    # get training data  
    reader = MatReader(TRAIN_PATH)  
    x_train = reader.read_field('sig')  
    x_train = x_train[:ntrain,::r_train[0],::r_train[1]][:,:s_train[0],:s_train[1]]  
    y_train = torch.stack([reader.read_field(key_map[i])\
    [:ntrain,::r_train[2],::r_train[3]][:,:s_train[2],:s_train[3]] for i in range(len(key_map))]).permute(1,2,3,0)  
    freq_base = reader.read_field('freq')[0]  
    obs_base = reader.read_field('obs')[0]  
    freq = torch.log10(freq_base[::r_train[2]][:s_train[2]])  
    obs = obs_base[::r_train[3]][:s_train[3]]/torch.max(obs_base)  
    nLoc = obs.shape[0]  
    nFreq = freq.shape[0]  
    freq = freq.view(nFreq, -1)  
    del reader  

    # get validation data  
    reader_val = MatReader(VAL_PATH)  
    x_val = reader_val.read_field('sig')  
    x_val = x_val[:nval,::r_val[0],::r_val[1]][:,:s_val[0],:s_val[1]]  
    y_val = torch.stack([reader_val.read_field(key_map[i])\
    [:nval,::r_val[2],::r_val[3]][:,:s_val[2],:s_val[3]] for i in range(len(key_map))]).permute(1,2,3,0)  
    del reader_val  

    # get test data  
    reader_test = MatReader(TEST_PATH)  
    x_test = reader_test.read_field('sig')  
    x_test = x_test[:ntest,::r_test[0],::r_test[1]][:,:s_test[0],:s_test[1]]  
    y_test = torch.stack([reader_test.read_field(key_map[i])\
    [:ntest,::r_test[2],::r_test[3]][:,:s_test[2],:s_test[3]] for i in range(len(key_map))]).permute(1,2,3,0)  
    del reader_test  

    # dataset normalization  
    x_normalizer = GaussianNormalizer(x_train)  
    x_train = x_normalizer.encode(x_train)  
    x_val = x_normalizer.encode(x_val)  
    x_test = x_normalizer.encode(x_test)  

    y_normalizer = GaussianNormalizer_out(y_train)  
    y_train = y_normalizer.encode(y_train)   

    # reshape data  
    x_train = x_train.reshape(ntrain, s_train[0], s_train[1], 1)  
    x_val = x_val.reshape(nval, s_val[0], s_val[1], 1)  
    x_test = x_test.reshape(ntest, s_test[0], s_test[1], 1)  

    return x_train, y_train, x_val, y_val, x_test, y_test, freq, nLoc, x_normalizer, y_normalizer

def print_model(model, flag=True):
    if flag:
        summary(model)

def get_warmup_learning_rate(epoch, base_lr, target_lr, warmup_epochs):
    """
    Compute the learning rate with warmup.
    Args:
        epoch: Current epoch (starting from 0)
        base_lr: Base learning rate (for single GPU)
        target_lr: Target learning rate (scaled for multi-GPU)
        warmup_epochs: Number of warmup epochs
    Returns:
        The learning rate for the current epoch
    """
    if epoch < warmup_epochs:
        # Linear warmup: Increase from base_lr to target_lr
        return base_lr + (target_lr - base_lr) * (epoch + 1) / warmup_epochs
    else:
        return target_lr

def get_distributed_training_config(world_size, base_batch_size, base_lr, epochs):
    """
    Dynamically adjust training parameters according to the number of GPUs, supporting 1 to 32+ GPUs.
    
    Args:
        world_size: Number of GPUs
        base_batch_size: The batch size for single GPU
        base_lr: Base learning rate
        epochs: Total number of training epochs
    
    Returns:
        dict: Contains batch_size, target_lr, warmup_epochs, etc.
    """
    import math
    
    if world_size == 1:
        # Default configuration for single GPU
        return {
            'batch_size': base_batch_size,
            'target_lr': base_lr,
            'warmup_epochs': 0
        }
    
    # Core logic for optimal parameter calculation
    def calculate_batch_size(world_size, base_batch_size):
        """Dynamically calculate per-GPU batch size."""
        if world_size <= 2:
            return base_batch_size
        elif world_size <= 4:
            # <=4 GPUs: moderately decrease batch size to keep efficiency
            return max(12, base_batch_size // 2)
        elif world_size <= 8:
            # 8 GPUs: balanced, avoid too large effective batch size
            return max(10, int(base_batch_size * 0.6))
        elif world_size <= 16:
            # 16 GPUs: more conservative to prevent overfitting
            return max(8, int(base_batch_size * 0.4))
        else:
            # 32+ GPUs: highly conservative, focus on convergence quality
            return max(6, int(base_batch_size * 0.25))
    
    def calculate_lr_scale(world_size):
        """Dynamically scale learning rate according to GPU count."""
        if world_size <= 2:
            return min(world_size * 0.7, 1.4)
        elif world_size <= 4:
            return 1.6
        elif world_size <= 8:
            return 1.8
        elif world_size <= 16:
            return 2.0
        else:
            # For large GPU count, use logarithmic scaling to prevent over-scaling
            return 2.0 + 0.3 * math.log2(world_size / 8)
    
    def calculate_warmup_ratio(world_size):
        """Dynamically set the warmup ratio (as portion of total epochs)."""
        if world_size <= 2:
            return 0.05
        elif world_size <= 4:
            return 0.05
        elif world_size <= 8:
            return 0.1
        elif world_size <= 16:
            return 0.15
        else:
            return 0.2  # Large scale training requires longer warmup

    batch_size = calculate_batch_size(world_size, base_batch_size)
    lr_scale = calculate_lr_scale(world_size)
    warmup_ratio = calculate_warmup_ratio(world_size)
    
    return {
        'batch_size': batch_size,
        'target_lr': base_lr * lr_scale,
        'warmup_epochs': max(1, int(epochs * warmup_ratio)),
        'effective_batch_size': batch_size * world_size,
        'lr_scale': lr_scale,
        'warmup_ratio': warmup_ratio
    }

def main(item):
    # Initialize the distributed environment. 
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  
    torch.cuda.set_device(local_rank)  

    if torch.cuda.is_available() is False:  
        raise EnvironmentError("not find GPU device for training.")
    
    dist.init_process_group(  
        backend='nccl',   
        init_method='env://',  # Use environment variables  
        world_size=int(os.environ.get('WORLD_SIZE', 1)),  
        rank=int(os.environ.get('RANK', 0))  
    )  

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Local Rank: {local_rank}, Global Rank: {rank}, World Size: {world_size}")
    dist.barrier()
    
    if rank == 0:
        t0 = default_timer()
        print(f'There are {world_size} GPUs in this training.')
    
    # load configurations from yaml file
    with open( 'config_EFDO.yml') as f:
        config = yaml.full_load(f)
    config = config[item]
    TRAIN_PATH = config['TRAIN_PATH']
    VAL_PATH   = config['VAL_PATH']
    TEST_PATH  = config['TEST_PATH']
    save_mode  = config['save_mode']
    save_step  = config['save_step']
    n_out      = config['n_out'] # rhoxy, rhoyx
    model_path = "../model/" + config['name'] + "_" + str(n_out) # save path and name of model
    model_path_temp = "../temp_model/" + config['name'] + "_" + str(n_out)
    log_path = "../log/" + config['name'] + "_" + str(n_out) + '.log'
    ntrain = config['ntrain']
    nval = config['nval']
    ntest  = config['ntest']
    
    epochs = config['epochs']
    train_config = get_distributed_training_config(
        world_size, 
        config['batch_size'], 
        config['learning_rate'], 
        epochs
    )

    batch_size = train_config['batch_size']
    base_learning_rate = config['learning_rate']
    target_learning_rate = train_config['target_lr']
    learning_rate = base_learning_rate
    warmup_epochs = train_config['warmup_epochs']

    step_size = config['step_size']
    gamma = config['gamma']
    modes = config['modes']
    width = config['width']
    s_train = config['s_train']
    r_train = config['r_train']
    s_val = config['s_val']
    r_val = config['r_val']
    s_test = config['s_test']
    r_test = config['r_test']
    layer_fno = config['layer_fno']
    layer_ufno = config['layer_ufno']
    layer_sizes = [s_train[0] * s_train[1]] + config['layer_sizes']
    act_fno   = config['act_fno']
    init_func = config['init_func']    
    patience = config['patience'] # if there is {patience} epoch that val_error is larger, early stop,
    thre_epoch = config['thre_epoch'] # condiser early stop after {thre_epoch} epochs
    print_model_flag = config['print_model_flag'] # print training model's structure and its parameters

    if rank == 0:
        print(f"=== Training Configuration ===")
        print(f"World size: {world_size}")
        print(f"Batch size: {batch_size}")
        
        if world_size == 1:
            # 单GPU训练配置
            print(f"Target learning rate: {base_learning_rate}")
            print(f"Warmup epochs: 0")
        else:
            # 分布式训练配置
            print(f"Per-GPU batch size: {batch_size}")
            print(f"Effective batch size: {train_config['effective_batch_size']}")
            print(f"Base learning rate: {base_learning_rate}")
            print(f"Target learning rate: {target_learning_rate:.6f}")
            print(f"Learning rate scale: {train_config['lr_scale']:.2f}x")
            print(f"Warmup epochs: {warmup_epochs}")
            print(f"Warmup ratio: {train_config['warmup_ratio']:.1%}")
        
        print(f"==============================")

    # Create directories if they don't exist (only rank 0 needs to do this)
    if rank == 0:
        os.makedirs("../model", exist_ok=True)
        os.makedirs("../temp_model", exist_ok=True)
        os.makedirs("../log", exist_ok=True)
        
    # load data and data normalization 
    if rank == 0:
        print("begin to load data")
        t_read0 = default_timer()

    x_train, y_train, x_val, y_val, x_test, y_test, freq, nLoc, _, y_normalizer = \
    get_batch_data(TRAIN_PATH, VAL_PATH, TEST_PATH, ntrain, nval, ntest, \
                   r_train, s_train,r_val, s_val,r_test,s_test,batch_size,n_out)
    
    dist.barrier()
    if rank == 0:
        t_read1 = default_timer()
        print(f"reading finished in {t_read1-t_read0:.3f} s")
    
    # pack training and testing data into torch.utils.data.TensorDataset
    train_data_set = torch.utils.data.TensorDataset(x_train, y_train)
    val_data_set = torch.utils.data.TensorDataset(x_val, y_val)
    test_data_set = torch.utils.data.TensorDataset(x_test, y_test)

    # allocate data to GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data_set, num_replicas=world_size, rank=rank, shuffle=False)  

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=4 )
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=test_sampler, pin_memory=True, num_workers=4 )

    freq = freq.cuda(local_rank)

    model = EFDO(modes, modes, width, n_out, layer_sizes, nLoc, init_func, layer_fno, layer_ufno, act_fno).cuda(local_rank) 

    ## using DistributedDataParallel to parallelize the model 
    ## model containing complex numbers cannot use DDP, but the pytorch 2.6.0 version can work well
    ## if using DDP, the manual gradient data sync can be omitted
    # model = torch.nn.parallel.DistributedDataParallel(  
    #     model,  
    #     device_ids=[local_rank],  
    #     output_device=local_rank  
    # )  

    # count model parameters
    if rank == 0:
        print_model(model, print_model_flag) 

    # setup the optimizer, learning decay, loss function
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    #----------------Using CosineAnnealingLR scheduler------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,  
        eta_min=target_learning_rate * 0.0005,  
        last_epoch=-1
    )
    if rank == 0:
        print(f"Using CosineAnnealingLR scheduler:")
        print(f"  T_max: {epochs - warmup_epochs}")
        print(f"  eta_min: {target_learning_rate * 0.01:.6f}")
    #---------------------------------------------------------------------
    myloss = LpLoss(size_average=False)

    # network training starts, and save the training log file
    if rank == 0:
        log_file = open(log_path,'a+')
        print("####################")
        print("begin to train model")  
        print("-" * 85)  
        print(f"{'Epoch':^10} | {'Time(s)':^12} | {'Train_Loss':^15} | {'Valid_Loss':^15} | {'Test_Loss':^15}")  
        print("-" * 85) 

    # training process   
    best_val_l2 = np.inf  
    stop_counter = 0  
    temp_file = None  
    best_epoch = 0  

    for epoch in range(epochs):  
        t1 = default_timer()  
        train_sampler.set_epoch(epoch)  

        # --- Warmup Phase ---
        if epoch < warmup_epochs:
            # Setup Warmup
            current_lr = get_warmup_learning_rate(epoch, base_learning_rate, target_learning_rate, warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler.step() 

        current_lr = optimizer.param_groups[0]['lr']
        # if rank == 0:
        #     print(f"Epoch {epoch+1}, Current LR: {current_lr_actual:.6f}")
        
        # Training phase  
        model.train()  
        train_l2 = 0.0  
        for data, target in train_loader:  
            data, target = data.cuda(rank), target.cuda(rank)  
            optimizer.zero_grad()  
            out = model(data, freq)  
            out = y_normalizer.decode(out)  
            target = y_normalizer.decode(target)  
            loss = myloss(out, target)  
            loss.backward()  
            
            # the higher torch version can omit this part if using DDP
            for param in model.parameters():  
                if param.grad is not None:  
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  
                    param.grad.data /= world_size  
                    
            optimizer.step()  
            train_l2 += loss.item()  
            
        # training loss  
        train_l2_tensor = torch.tensor(train_l2).to(rank)  
        dist.all_reduce(train_l2_tensor, op=dist.ReduceOp.SUM)  
        train_l2 = train_l2_tensor.item()  
        train_l2 /= ntrain   
        
        # if epoch >= warmup_epochs:
        #     scheduler.step() 
        #     


        # Validation phase  
        model.eval()  
        val_l2 = 0.0  
        with torch.no_grad():  
            for data, target in val_loader:  
                data, target = data.cuda(rank), target.cuda(rank)  
                out = model(data, freq)  
                out = y_normalizer.decode(out)  
                val_l2 += myloss(out, target).item()  
                
        # validation loss 
        val_l2_tensor = torch.tensor(val_l2).to(rank)  
        dist.all_reduce(val_l2_tensor, op=dist.ReduceOp.SUM)  
        val_l2 = val_l2_tensor.item()  
        val_l2 /= nval  

        # Testing phase  
        test_l2 = 0.0  
        with torch.no_grad():  
            for data, target in test_loader:  
                data, target = data.cuda(rank), target.cuda(rank)  
                out = model(data, freq)  
                out = y_normalizer.decode(out)  
                test_l2 += myloss(out, target).item()  
                
        # testing loss  
        test_l2_tensor = torch.tensor(test_l2).to(rank)  
        dist.all_reduce(test_l2_tensor, op=dist.ReduceOp.SUM)  
        test_l2 = test_l2_tensor.item()  
        test_l2 /= ntest  

        # save model  
        if (epoch+1) % save_step == 0:  
            if rank == 0:  
                # if temp_file is not None:  
                #     os.remove(temp_file)
                torch.save(model.state_dict(), model_path_temp + '_epoch_' + str(epoch+1) + '.pkl')  
                # temp_file = model_path_temp + '_epoch_' + str(epoch + 1) + '.pkl'    

        # early stop  
        if (epoch+1) > thre_epoch:  
            if val_l2 < best_val_l2:  
                best_val_l2 = val_l2  
                best_epoch = epoch  
                stop_counter = 0   
                if rank == 0:  
                    if save_mode == 'state_dict':  
                        torch.save(model.state_dict(), model_path + '_epoch_' + str(epoch+1) + '.pkl')  
                    else:  
                        torch.save(model, model_path + '_epoch_' + str(epoch+1) + '.pt')  
            else:  
                stop_counter += 1  
            if stop_counter > patience:  
                if rank == 0:  
                    print(f"Early stop at epoch {epoch}")  
                    print(f"Best model was saved at epoch {best_epoch+1}")  
                    print(f"# Early stop at epoch {epoch}", file=log_file)  
                    print(f"# Best model was saved at epoch {best_epoch+1}", file=log_file)  
                break  
        
        t2 = default_timer()  
        if rank == 0:  
            print(epoch + 1, t2 - t1, train_l2, val_l2, test_l2, current_lr)  
            print(epoch + 1, t2 - t1, train_l2, val_l2, test_l2, file = log_file)

    if rank == 0:
        tn = default_timer()
        print(f'all time:{tn-t0:.3f}s')
        print(f'# all time:{tn-t0:.3f}s',file=log_file)
        log_file.close()

    cleanup()    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--item', default='EFDO_config', help='config_EFDO.yml')  

    args = parser.parse_args() 
    # print("Current args: ", vars(args)) 
    item = args.item  
    main(item)  
    
# python -m torch.distributed.run --nproc_per_node=4 EFDO_main_parallel.py --item EFDO_config