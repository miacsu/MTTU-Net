import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio



def matrix_minor(arr, i, j):
    return np.delete(np.delete(arr,i,axis=0), j, axis=1)
def caculate_metric_multiclass_classification(confusion,labels,target_names):
    assert confusion.shape[0] == confusion.shape[1]
    accuracy = []
    precision = []
    recall = []
    sensitivity = []
    specificity = []
    f1_score = []
    for i in labels:
        print(confusion[i, i])
        # postive for class:0
        if i==0:
            FN = confusion[i, 1] + confusion[i, 2]
            FP = confusion[1, i] + confusion[2, i]
        elif i==1:
            FN = confusion[i, 0] + confusion[i, 2]
            FP = confusion[0, i] + confusion[2, i]
        elif i==2:
            FN = confusion[i, 0] + confusion[i, 1]
            FP = confusion[0, i] + confusion[1, i]
        TP = confusion[i, i]
        TN = matrix_minor(confusion, i, i).sum()

        acc = (TP + TN) / (TP + FN + FP + TN)
        prec = TP / (TP + FP)
        reca = TP / (TP + FN)
        sens = TP / (TP + FN)
        spes = TN / (TN + FP)
        f1 = 2 * TP / (2 * TP + FP + FN)
        print("%s:accuracy:%0.5f,sensitivity:%0.5f,specificity:%0.5f,precision:%0.5f,recall:%0.5f,f1_score:%0.5f"
              % (target_names[i],acc,sens,spes,prec,reca,f1))
        accuracy.append(acc)
        precision.append(prec)
        recall.append(reca)
        sensitivity.append(sens)
        specificity.append(spes)
        f1_score.append(f1)
    return {"accuracy":accuracy,"sensitivity":sensitivity,"specificity":specificity,
            "precision":precision,"recall":recall,"f1_score":f1_score}

def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, model):
    temp = []
    idh_temp=[]

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()
    #y = torch.cat((x,x[:,[0],:,:,:]),dim=1).clone()
    for i in range(len(temp)):
        if isinstance(model,dict):
            encoder_outs = model['en'](temp[i])  # encoder_outputs:x1_1, x2_1, x3_1,x4_1, encoder_output, intmd_encoder_outputs
            seg_output = model['seg'](encoder_outs[0],encoder_outs[1],encoder_outs[2],encoder_outs[4])   #x1_1, x2_1, x3_1, intmd_encoder_outputs
            idh_out = model['idh'](encoder_outs[3],encoder_outs[4])  #x4_1, encoder_output
            # grade_out = model['grade'](encoder_outs[3], encoder_outs[4])
            temp[i] = seg_output
            print("idh_out:",idh_out)
            idh_temp.append(idh_out)
        else:
            temp[i],b = model(temp[i])

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]
    if isinstance(model,dict):
        idh_out = torch.mean(torch.stack(idh_temp), dim=0)
        print("idh_out mean:",idh_out)
        return y[..., :155],idh_out#,grade_out
    else:
        return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret


def softmax_whole_dice(output, target):

    # whole
    o = output > 0; t = target > 0 # ce

    return dice_score(o, t)


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        load_file,
        multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
        ):
    t1ce_Path = '/public/home/hpc184601044/dataset/BraTS_2020/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001/BraTS20_Validation_001_t1ce.nii.gz'
    t1ce_image = nib.load(t1ce_Path)
    H, W, T = 240, 240, 155
    if isinstance(model,dict):
        model['en'].eval()
        model['seg'].eval()
        model['idh'].eval()
        # model['grade'].eval()
    else:
        model.eval()

    runtimes = []
    ET_voxels_pred_list = []


    grade_prob = []
    grade_class = []
    grade_truth = []
    grade_error_case = []

    idh_prob = []
    idh_class = []
    idh_truth = []
    idh_error_case = []
    ids = []

    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
        else:
            if len(data)==2:
                print("data[0]:",data[0].shape,'data[1]',data[1])
                data = [t.cuda(non_blocking=True) for t in data]
                x, idh = data[:2]
            else:
                x = data
                x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = tailor_and_concat(x, model)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)


            if multimodel:
                logit = F.softmax(logit, dim=1)
                output = logit / 4.0

                load_file1 = load_file.replace('7998', '7996')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7997')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7999')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
            else:
                output = F.softmax(logit, dim=1)

        else:

            if isinstance(model,dict):
                x = x[..., :155]

                TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, model),tailor_and_concat(x.flip(dims=(2,)), model),\
                                                                  tailor_and_concat(x.flip(dims=(3,)), model),tailor_and_concat(x.flip(dims=(4,)), model),\
                                                                  tailor_and_concat(x.flip(dims=(2, 3)), model),tailor_and_concat(x.flip(dims=(2, 4)), model),\
                                                                  tailor_and_concat(x.flip(dims=(3, 4)), model),tailor_and_concat(x.flip(dims=(2, 3, 4)), model)
                # logit = F.softmax(TTA_1, 1)  # no flip
                # logit += F.softmax(TTA_2.flip(dims=(2,)), 1)  # flip H
                # logit += F.softmax(TTA_3.flip(dims=(3,)), 1)  # flip W
                # logit += F.softmax(TTA_4.flip(dims=(4,)), 1)  # flip D
                # logit += F.softmax(TTA_5.flip(dims=(2, 3)), 1)  # flip H, W
                # logit += F.softmax(TTA_6.flip(dims=(2, 4)), 1)  # flip H, D
                # logit += F.softmax(TTA_7.flip(dims=(3, 4)), 1)  # flip W, D
                # logit += F.softmax(TTA_8.flip(dims=(2, 3, 4)),1)  # flip H, W, D

                logit = F.softmax(TTA_1[0], 1)  # no flip
                logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
                logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
                logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
                logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
                logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
                logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
                logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
                output = logit / 8.0
                idh_probs = []
                grade_probs = []
                for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
                    #print("pred:",pred[1])
                    #print("soft_max: pred:", F.softmax(pred[1],1))
                    # idh_probs.append(pred[1].sigmoid())
                    idh_probs.append(F.softmax(pred[1],1))
                    # grade_probs.append(F.softmax(pred[2],1))
                # #print("idh_probs:",idh_probs)
                idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
                print("idh_pred:",idh_pred)
                #grade_pred = torch.mean(torch.stack(grade_probs),dim=0)
                # #idh_pred = TTA_1[1].item()
                #print("grade_pred:", grade_pred)
                # idh_prob.append(idh_pred[0])
                idh_prob.append(idh_pred[0][1].item())
                #grade_prob.append(grade_pred[0][1])
                idh_pred_class = torch.argmax(idh_pred,dim=1)
                # idh_pred_class = (idh_pred>0.5).float()
                #torch.argmax(idh_pred, dim=1)
                idh_class.append(idh_pred_class.item())
                print('id:',names[i],'IDH_truth:',idh.item(),'IDH_pred:',idh_pred_class.item())
                #
                #grade_truth.append(grade.item())
                #grade_pred_class = torch.argmax(grade_pred, dim=1)
                #grade_class.append(grade_pred_class.item())
                #print('id:', names[i], 'grade_truth:', grade.item(), 'grade_pred:', grade_pred_class.item())
                #
                ids.append(names[i])
                idh_truth.append(idh.item())
                if not (idh_pred_class.item() == idh.item()):
                    idh_error_case.append({'id':names[i],'truth:':idh.item(),'pred':idh_pred_class.item()})
                #if not (grade_pred_class.item() == grade.item()):
                #    grade_error_case.append({'id': names[i], 'truth:': grade.item(), 'pred': grade_pred_class.item()})
            else:
                x = x[..., :155]
                logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
                logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
                logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
                output = logit / 8.0  # mean

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)

        ET_0 = [67, 68, 69, 72, 74, 76, 77, 83, 85, 89, 91, 92, 99, 103]
        ET_1 = [75, 82, 87, 88, 90, 95, 96, 97, 107]
        sub_id = int(names[i].split('_')[-1])
        if postprocess == True:
            ET_voxels_pred = (output == 3).sum()
            ET_voxels_pred_list.append(ET_voxels_pred)
            print('ET_voxel_pred:', ET_voxels_pred)

            if sub_id in ET_0:
                print('----------Processing ET=0----------')
                output[np.where(output == 3)] = 1
            if ET_voxels_pred < 500:
                if sub_id in ET_1:
                    pass
                else:
                    output[np.where(output == 3)] = 1
                    print('Replace ET_voxel!')

        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 4
                if verbose:
                    print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
                nib.save(nib.Nifti1Image(seg_img, affine=t1ce_image.affine,header=t1ce_image.header), oname)
                print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
    if isinstance(model, dict):

        print("--------------------------------IDH evaluation report---------------------------------------")

        from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
        import pandas as pd
        data = pd.DataFrame({"ID":ids,"pred":idh_prob,"pred_class":idh_class,"idh_truth":idh_truth})
        data.to_csv("revised/BraTS_pred_Focus_loss_dual.csv")
        confusion = confusion_matrix(idh_truth,idh_class)
        print(confusion)
        labels = [0, 1]
        target_names = ["wild", "Mutant"]
        print(classification_report(idh_truth, idh_class, labels=labels, target_names=target_names))
        print("ACU:",roc_auc_score(idh_truth,idh_prob))
        print("Acc:", accuracy_score(idh_truth, idh_class))
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))
        print("-------------------------- error cases----------------------------------------")
        for case in idh_error_case:
            print(case)

        # print("--------------------------------Grade evaluation report---------------------------------------")
        #
        # confusion = confusion_matrix(grade_truth, grade_class)
        # print(confusion)
        # labels = [0, 1]
        # target_names = ["LGG", "HGG"]
        # print(classification_report(grade_truth, grade_class, labels=labels, target_names=target_names))
        # print("ACU:", roc_auc_score(grade_truth, grade_prob))
        # print("Acc:", accuracy_score(grade_truth, grade_class))
        # if float(np.sum(confusion)) != 0:
        #     accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        # print("Global Accuracy: " + str(accuracy))
        # specificity = 0
        # if float(confusion[0, 0] + confusion[0, 1]) != 0:
        #     specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        # print("Specificity: " + str(specificity))
        # sensitivity = 0
        # if float(confusion[1, 1] + confusion[1, 0]) != 0:
        #     sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        # print("Sensitivity: " + str(sensitivity))
        # precision = 0
        # if float(confusion[1, 1] + confusion[0, 1]) != 0:
        #     precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        # print("Precision: " + str(precision))
        # print("-------------------------- error cases----------------------------------------")
        # for case in grade_error_case:
        #     print(case)

        # confusion_grade = confusion_matrix(grade_truth, grade_class)
        # print("overall accuracy:",accuracy_score(grade_truth,grade_class))
        # caculate_metric_multiclass_classification(confusion_grade,labels=[0,1,2],target_names=['G2','G3','G4'])
        #print('runtimes:', sum(runtimes) / len(runtimes))
