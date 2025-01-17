from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, Softmax, Normface, ArcfaceOrigin, ArcfaceOriginAdaptiveM, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz

class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).cuda()
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).cuda()
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        self.model = torch.nn.DataParallel(self.model).cuda()
        
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)
            print('Total batch is {}'.format(len(self.loader)))
            print('class_num is {}'.format(self.class_num))
            self.writer = SummaryWriter(conf.log_path, max_queue=20)
            self.step = 0
            # self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            if conf.head == 'softmax':
                self.head = Softmax(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            elif conf.head == 'normface':
                self.head = Normface(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            elif conf.head == 'normface_alter-grad':
                self.head = Normface(embedding_size=conf.embedding_size, classnum=self.class_num, alter_grad=True).cuda()
            elif conf.head == 'arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            elif conf.head == 'arcface_alter-grad':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num, alter_grad=True).cuda()
            elif conf.head == 'arcface_origin':
                self.head = ArcfaceOrigin(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            elif conf.head == 'arcface_origin_detach-diff':
                self.head = ArcfaceOrigin(embedding_size=conf.embedding_size, classnum=self.class_num, detach_diff=True).cuda()
            elif conf.head == 'arcface_adaptivemargin':
                self.head = ArcfaceOriginAdaptiveM(embedding_size=conf.embedding_size, classnum=self.class_num, m=conf.margin,
                                          detach_diff=conf.detach_diff, m_mode=conf.m_mode).cuda()

            self.head = torch.nn.DataParallel(self.head).cuda()

            print('head:')
            print(self.head)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.module)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.module.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        if not save_path.exists():
            save_path.mkdir()
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('Evaluate/{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('Evaluate/{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('Evaluate/{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, dataset, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    emb_batch = self.model(batch.cuda()).cpu()
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                idx += conf.batch_size

                # tensorboard feature length during testing
                if idx + conf.batch_size > len(carray):
                    # the length of features
                    feature_norms = emb_batch.data.norm(p=2, dim=1)
                    self.writer.add_histogram('Feature_Length/test/' + dataset, feature_norms, self.step)
                    self.writer.add_scalar('Mean_Feature_Length/test/' + dataset, feature_norms.mean(), self.step)

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    emb_batch = self.model(batch.cuda()).cpu()
                    embeddings[idx:] = l2_norm(emb_batch)
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)

        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = l2_norm(self.model(imgs))
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
            # adjust learning rate
            if e in self.milestones:
                self.schedule_lr()

            # board learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', lr, e+1)

            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.cuda()
                labels = labels.cuda()
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas, cos_thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0:
                    # tensorboard train loss
                    if self.step == 0:
                        loss_board = running_loss
                    else:
                        loss_board = running_loss / self.board_loss_every
                        running_loss = 0.
                    self.writer.add_scalar('train_loss', loss_board, self.step)

                    # tensorboard weights cosine and weight norm in self.head
                    mean_cosine, max_cosine, weight_norm = self.compute_weight_cosine()
                    self.writer.add_scalar('Weights_Cosine/mean_cosine', mean_cosine, self.step)
                    self.writer.add_scalar('Weights_Cosine/max_cosine', max_cosine, self.step)
                    self.writer.add_histogram('Weight_Length', weight_norm, self.step)
                    self.writer.add_scalar('Weight_Length_Mean', weight_norm.mean(), self.step)

                    # the length of features
                    feature_norms = embeddings.data.norm(p=2, dim=1)
                    self.writer.add_histogram('Feature_Length/train', feature_norms, self.step)
                    self.writer.add_scalar('Mean_Feature_Length/train', feature_norms.mean(), self.step)

                    # the cos_thetas in the head
                    self.writer.add_histogram('Cos_Thetas/train', cos_thetas, self.step)
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame, 'agedb_30')
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame, 'lfw')
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame, 'cfp_fp')
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = l2_norm(self.model(conf.test_transform(img).cuda().unsqueeze(0)))
                emb_mirror = l2_norm(self.model(conf.test_transform(mirror).cuda().unsqueeze(0)))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(l2_norm(self.model(conf.test_transform(img).cuda().unsqueeze(0))))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum

    def compute_weight_cosine(self):
        weight = self.head.module.kernel.data.cpu()
        weight_norm = weight.norm(dim=0, keepdim=True)
        weight = weight / weight_norm

        cosine_similarity = torch.matmul(weight.t(), weight).tril(diagonal=-1)
        mean_cosine_similarity = cosine_similarity.sum() / ((self.class_num * (self.class_num - 1.0)) / 2.0)

        return mean_cosine_similarity, cosine_similarity.max(), weight_norm

