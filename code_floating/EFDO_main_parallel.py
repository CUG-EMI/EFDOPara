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
os.environ["OMP_NUM_THREADS"] = "1"    
os.environ["MKL_NUM_THREADS"] = "1"    

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

def main(item):
    # 1. Initialize the distributed environment. 
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  
    torch.cuda.set_device(local_rank)  

    if torch.cuda.is_available() is False:  
        raise EnvironmentError("not find GPU device for training.")
    
    # dist.init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}')
    # dist.init_process_group(  
    #     backend='nccl',   
    #     init_method='env://',  # Use environment variables  
    #     world_size=int(os.environ.get('WORLD_SIZE', 1)),  
    #     rank=int(os.environ.get('RANK', 0))  
    # )  
    rank = int(os.environ["RANK"])  
    world_size = int(os.environ["WORLD_SIZE"])  
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")  
    master_port = os.environ.get("MASTER_PORT", "29504")  

    dist.init_process_group(  
        backend='nccl',  
        init_method=f'tcp://{master_addr}:{master_port}',  
        rank=rank,  
        world_size=world_size  
    )

    rank = dist.get_rank()
    # torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    print(f"Local Rank: {local_rank}, Global Rank: {rank}, World Size: {world_size}")
    
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
    model_path_temp = "../temp_model/Par" + config['name'] + "_" + str(n_out)
    log_path = "../log/" + config['name'] + "_" + str(n_out) + '.log'
    ntrain = config['ntrain']
    nval = config['nval']
    ntest  = config['ntest']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
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

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=16, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=16, drop_last=True, persistent_workers=True )
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=test_sampler, pin_memory=True, num_workers=16, drop_last=True, persistent_workers=True )

    freq = freq.cuda(local_rank)

    model = EFDO(modes, modes, width, n_out, layer_sizes, nLoc, init_func, layer_fno, layer_ufno, act_fno).cuda(local_rank) 

    ## using DistributedDataParallel to parallelize the model 
    ## model containing complex numbers cannot use DDP, but the pytorch 2.6.0 version can work well
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
            
            # if model containing complex numbers, use the following code to maunual average gradients 
            # the higher torch version can omit this part
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
        
        scheduler.step()  

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
            print(epoch + 1, t2 - t1, train_l2, val_l2, test_l2)  
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
    # parser.add_argument('--local_rank', type=int, default=0) 
    # parser.add_argument('--node_rank', type=int, default=0)  
    # parser.add_argument('--master_addr', type=str, default='')  
    # parser.add_argument('--master_port', type=int, default=0)  

    args = parser.parse_args() 
    # print("Current args: ", vars(args)) 
    item = args.item  
    main(item)  
    
# python -m torch.distributed.run --nproc_per_node=4 EFDO_main_parallel.py --item EFDO_confi