import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from model import Coma
from transform import Normalize
from psbody.mesh import Mesh
from utils import get_vert_connectivity
from read_obj import save_obj
import glob


class ComaDataset(InMemoryDataset):
    def __init__(self, root_dir, dtype='train', split='sliced', split_term='sliced', nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_tranform = pre_transform
        # Downloaded data is present in following format root_dir/*/*/*.py
        self.data_file = glob.glob(self.root_dir + '/*/*/*.obj')

        super(ComaDataset, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")


        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
        processed_files = [self.split_term+'_'+pf for pf in processed_files]
        return processed_files

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        count = 0
        for idx, data_file in tqdm(enumerate(self.data_file)):
            fn = data_file.split('/')
            #print(fn[2])
            mesh = Mesh(filename=data_file)
            mesh_verts = torch.Tensor(mesh.v)
           # print(torch.mean(mesh_verts, dim = 0, keepdim = True))
            #mesh_verts = mesh_verts-torch.mean(mesh_verts, dim = 0, keepdim = True)
            '''
            if idx % 100 == 0:
                print(torch.mean(mesh_verts, dim=0))
                save_obj(mesh_verts, mesh.f+1, os.path.join('./vis/', 'data'+str(idx)+'.obj'))
            '''

            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col))).long()
          
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

            if self.split == 'sliced':
                if idx % 100 <= 10:
                    test_data.append(data)
               # elif idx % 100 <= 10:
                #    val_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'expression':
                if data_file.split('/')[-2] == self.split_term:
                    test_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'identity':
                if data_file.split('/')[-3] == self.split_term:
                    test_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)
            else:
                raise Exception('sliced, expression and identity are the only supported split terms')

        if self.split != 'sliced':
            val_data = test_data[-self.nVal:]
            test_data = test_data[:-self.nVal]

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        print(mean_train.shape)
        return
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(test_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])

def prepare_sliced_dataset(path):
    ComaDataset(path, pre_transform=Normalize())


def prepare_expression_dataset(path):
    test_exps = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up', 'mouth_down',
                 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
    for exp in test_exps:
        ComaDataset(path, split='expression', split_term=exp, pre_transform=Normalize())

def prepare_identity_dataset(path):
    test_ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170811_03274_TA',
                'FaceTalk_170904_00128_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170913_03279_TA',
                'FaceTalk_170728_03272_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170811_03275_TA',
                'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170915_00223_TA']

    for ids in test_ids:
        ComaDataset(path, split='identity', split_term=ids, pre_transform=Normalize())



def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    visualize = config['visualize']
    output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    normalize_transform = Normalize()

    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.to(device)
    print('making...')
    norm = torch.load('../processed_data/processed/sliced_norm.pt')
    normalize_transform.mean = norm['mean']
    normalize_transform.std = norm['std']
 
    #'0512','0901','0516','0509','0507','9305','0503','4919','4902',
    files = ['0514','0503', '0507', '0509' ,'0512', '0501', '0901',  '1001', '4902', '4913', '4919','9302', '9305', '12411']

    coma.eval()
    
    meshviewer = MeshViewers(shape=(1, 2))
    for file in files:
        #mat = np.load('../Dress Dataset/'+file+'/'+file+'_pose.npz')
        mesh_dir = os.listdir('../processed_data/'+file+'/mesh/')
        latent = []
        print(len(mesh_dir))
        for i in tqdm(range(len(mesh_dir))):
            data_file = '../processed_data/'+file+'/mesh/'+str(i)+'.obj'
            mesh = Mesh(filename=data_file)
            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col))).long()
            mesh_verts = (torch.Tensor(mesh.v)-normalize_transform.mean)/normalize_transform.std
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
            data = data.to(device)
            with torch.no_grad():
                out, feature = coma(data)
                latent.append(feature.cpu().detach().numpy())
               # print(feature.shape)
            if i % 50 ==0:
                expected_out = data.x
                out = out.cpu().detach()*normalize_transform.std + normalize_transform.mean
                expected_out = expected_out.cpu().detach()*normalize_transform.std + normalize_transform.mean
                out = out.numpy()
                save_obj(out, template_mesh.f+1, './vis/reconstruct_'+str(i)+'.obj')
                save_obj(expected_out, template_mesh.f+1, './vis/ori_'+str(i)+'.obj')

        np.save('./processed/0820/'+file, latent)

    




    if torch.cuda.is_available():
        torch.cuda.synchronize()




def evaluate(coma, output_dir, test_loader, dataset, template_mesh, device, visualize=False):
    coma.eval()
    latent = []
    total_loss = 0
    meshviewer = MeshViewers(shape=(1, 2))
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data).detach().cpu().numpy()
            latent.append(out)

  
    return total_loss/len(dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
