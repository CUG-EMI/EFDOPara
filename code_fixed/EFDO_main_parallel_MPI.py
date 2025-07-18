import os  
import argparse
import numpy as np  
import mpi4py.MPI as MPI  
import torch  
import torch.distributed as dist  
from torchinfo import summary  
import yaml  
from timeit import default_timer  
from utilities import *  
from EFDO import EFDO  

torch.manual_seed(0)  
np.random.seed(0)  

def initialize_environment():  
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()  
    size = comm.Get_size()  
    
    # setting the device for each process 
    torch.cuda.set_device(rank)  
    # initialize the distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend='nccl', world_size=size, rank=rank)  
    
    return comm, rank, size

# def initialize_environment():
#     import random
#     import os
    
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
    
#     # 如果没有设置MASTER_PORT环境变量，随机生成一个
#     if 'MASTER_PORT' not in os.environ:
#         if rank == 0:
#             port = random.randint(10000, 65000)
#             os.environ['MASTER_PORT'] = str(port)
#             print(f"Using random port: {port}")
#         else:
#             # 非主进程等待主进程设置端口
#             port = None
            
#         # 广播端口号给所有进程
#         port = comm.bcast(port, root=0)
#         if rank != 0:
#             os.environ['MASTER_PORT'] = str(port)
    
#     # 确保设置MASTER_ADDR
#     if 'MASTER_ADDR' not in os.environ:
#         os.environ['MASTER_ADDR'] = 'localhost'
        
#     # 打印当前使用的端口和地址
#     if rank == 0:
#         print(f"Using MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
#         print(f"Using MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
#     # 尝试初始化进程组，如果失败则重试不同端口
#     try:
#         dist.init_process_group(backend='nccl', world_size=size, rank=rank)
#     except RuntimeError as e:
#         if "EADDRINUSE" in str(e) and rank == 0:
#             print(f"Port {os.environ.get('MASTER_PORT')} already in use, trying another...")
#             port = random.randint(10000, 65000)
#             os.environ['MASTER_PORT'] = str(port)
#             print(f"Trying new port: {port}")
        
#         # 广播新端口号
#         port = comm.bcast(int(os.environ.get('MASTER_PORT', 29500)), root=0)
#         if rank != 0:
#             os.environ['MASTER_PORT'] = str(port)
        
#         # 同步所有进程后再次尝试初始化
#         comm.Barrier()
#         dist.init_process_group(backend='nccl', world_size=size, rank=rank)
    
#     return comm, rank, size

def cleanup():
    dist.destroy_process_group()

def load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, ntrain, nval, ntest,   
              r_train, s_train, r_val, s_val, r_test, s_test,   
              batch_size, n_out, rank, size):  
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

    # allocate data to each process
    x_train = x_train[rank::size]  
    y_train = y_train[rank::size]  
    x_val = x_val[rank::size]  
    y_val = y_val[rank::size]  
    x_test = x_test[rank::size]  
    y_test = y_test[rank::size]  

    # create TensorDataset 
    train_data_set = torch.utils.data.TensorDataset(x_train, y_train)  
    val_data_set = torch.utils.data.TensorDataset(x_val, y_val)  
    test_data_set = torch.utils.data.TensorDataset(x_test, y_test)  

    # using DataLoader to load data 
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,   
                                             pin_memory=True, num_workers=1,   
                                             shuffle=True, drop_last=True)  
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size,   
                                           pin_memory=True, num_workers=1)  
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size,   
                                            pin_memory=True, num_workers=1)  

    return train_loader, val_loader, test_loader, freq, nLoc, x_normalizer, y_normalizer  

def print_model(model, flag=True):  
    if flag:  
        summary(model)  

def main(item):  
    if torch.cuda.is_available() is False:  
        raise EnvironmentError("not find GPU device for training.")  
    
    comm, rank, size = initialize_environment()  
    
    if rank == 0:  
        t0 = default_timer()  
        print(f'There are {size} GPUs in this training.')  
    
    # load configurations from yaml file  
    with open('config_EFDO.yml') as f:  
        config = yaml.full_load(f)  
    config = config[item]  
    TRAIN_PATH = config['TRAIN_PATH']  
    VAL_PATH = config['VAL_PATH']  
    TEST_PATH = config['TEST_PATH']  
    save_mode = config['save_mode']  
    save_step = config['save_step']  
    n_out = config['n_out']  
    model_path = "../model/" + config['name'] + "_" + str(n_out)  
    model_path_temp = "../temp_model/MPI" + config['name'] + "_" + str(n_out)  
    log_path = "../log/" + config['name'] + "_" + str(n_out) + '.log'  
    ntrain = config['ntrain']  
    nval = config['nval']  
    ntest = config['ntest']  
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
    act_fno = config['act_fno']  
    init_func = config['init_func']  
    patience = config['patience']  
    thre_epoch = config['thre_epoch']  
    print_model_flag = config['print_model_flag']  

    # Create directories if they don't exist (only rank 0 needs to do this)
    if rank == 0:
        os.makedirs("../model", exist_ok=True)
        os.makedirs("../temp_model", exist_ok=True)
        os.makedirs("../log", exist_ok=True)
        
    if rank == 0:  
        print("begin to load data")  
        t_read0 = default_timer()  

    train_loader, val_loader, test_loader, freq, nLoc, _, y_normalizer = \
    load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, ntrain, nval, ntest,  
              r_train, s_train, r_val, s_val, r_test, s_test,  
              batch_size, n_out, rank, size)  

    if rank == 0:  
        t_read1 = default_timer()  
        print(f"reading finished in {t_read1-t_read0:.3f} s")  

    freq = freq.cuda()  
    model = EFDO(modes, modes, width, n_out, layer_sizes, nLoc,   
                 init_func, layer_fno, layer_ufno, act_fno).cuda()  

    # broadcast model parameters from rank 0 to all other ranks
    for param in model.parameters():  
        param_cpu = param.data.cpu().numpy()  
        comm.Bcast(param_cpu, root=0)  
        param.data.copy_(torch.from_numpy(param_cpu).cuda())  

    if rank == 0:  
        print_model(model, print_model_flag)  

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  
    myloss = LpLoss(size_average=False)  

    if rank == 0:  
        log_file = open(log_path,'a+')  
        print("####################")  
        print("begin to train model")  
        print("-" * 85)  
        print(f"{'Epoch':^10} | {'Time(s)':^12} | {'Train_Loss':^15} | {'Valid_Loss':^15} | {'Test_Loss':^15}")  
        print("-" * 85) 

    best_val_l2 = np.inf  
    stop_counter = 0  
    temp_file = None  
    best_epoch = 0  

    for epoch in range(epochs):  
        t1 = default_timer()  
        model.train()  
        train_l2 = 0.0  
        for data, target in train_loader:  
            data, target = data.cuda(), target.cuda()  
            optimizer.zero_grad()  
            out = model(data, freq)  
            out = y_normalizer.decode(out)  
            target = y_normalizer.decode(target)  
            loss = myloss(out, target)  
            loss.backward()  

            for param in model.parameters():  
                if param.grad is not None:  
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  
                    param.grad.data /= size   

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
                data, target = data.cuda(), target.cuda()  
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
                data, target = data.cuda(), target.cuda()  
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
                # temp_file = model_path_temp + '_epoch_' + str(epoch+1) + '.pkl'  

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
            print(epoch + 1, t2 - t1, train_l2, val_l2, test_l2, file=log_file)  

    if rank == 0:  
        tn = default_timer()  
        print(f'all time:{tn-t0:.3f}s')  
        print(f'# all time:{tn-t0:.3f}s', file=log_file)  
        log_file.close()  
    
    cleanup() 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--item', default='EFDO_config', help='config_EFDO.yml')   
    args = parser.parse_args()
    item = args.item  
    main(item)

# mpirun -np 2 python EFDO_main_parallel_MPI.py