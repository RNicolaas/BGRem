###################
# bgrem_gamma-noPoisson.py saves the best model and 
# obtained bgrem images on the test set.
# Here we use SourceExtractor for detecting sources on bgrem images
# and compare the results with the groundtruth and 
# the default SourceExtractor background removal method.
###################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import sep
import os


home_path = "/d11/CAC/sbhattacharyya/Downloads/Rodney_Bgrem/"
checkpoint_path = home_path + "checkpoints/"
bgrem_path = checkpoint_path + 'bg_removed_LAT_test/'

im_path = home_path + 'test_im_iem_psr_bll_fsrq_pwn1_2_patch768_tbin1/'

image_size = 64


def extract_sources(image, im_name, save_path, type_im, bgfactor=3,):
    '''
    extract sources and draw ellipses around them
    use sep bkg. estimator
    find srcs that are above bgfactor x bkg_rms above bkg
    bgfactor =3; i.e. find blobs that are 3\sigma above bkg noise
    plot ellipses around found sources (use mpl ellipse)
    '''
    im_id = im_name.split('_')[2].split('.')[0]
    bkg = sep.Background(image)
    objects = sep.extract(image, bgfactor, err=bkg.globalrms)
    print('Sources found: '+str(len(objects)))
    fig, ax = plt.subplots()
    # m, s = np.mean(image), np.std(image)/20
    # image = image +  1e-9
    im = ax.imshow(np.sqrt(image), cmap='inferno', origin='lower')

    
    # plot ellipse for every obj.
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]), 
                        width=6*objects['a'][i], 
                        height=6*objects['b'][i], 
                        angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        if type_im !='iem_src': # don't add ellipses on the bkg_src image
            ax.add_artist(e)
        fig.savefig(save_path + f'{type_im}_{im_id}_withSq.png', dpi=200, bbox_inches='tight')
        plt.close(fig)    


def read_data(bgrem_path=bgrem_path, org_path = im_path):
    image_names = os.listdir(bgrem_path)
     
    bgrem_images = np.zeros((len(image_names), image_size, image_size, 1))
    original_images = np.zeros((len(image_names), image_size, image_size, 1))
    source_images = np.zeros((len(image_names), image_size, image_size, 1))
    for i in range(len(image_names)):
        org_image_all_comp = np.load(org_path + image_names[i])
        bgrem_images[i] = np.load(bgrem_path + image_names[i])
        
        original_images[i] = (org_image_all_comp[0] + 
                              org_image_all_comp[1] + 
                              org_image_all_comp[2] + 
                              org_image_all_comp[3] + 
                              org_image_all_comp[4])
        source_images[i] = (org_image_all_comp[1] + 
                            org_image_all_comp[2] + 
                            org_image_all_comp[3] + 
                            org_image_all_comp[4])
    original_images *= 10 # from yearly photon counts to 10 yrs data
    source_images *= 10
    # original_images += np.random.poisson(original_images) - original_images
    return(bgrem_images, original_images, source_images, image_names)

bgrem_ims, org_ims, src_ims, im_names = read_data(bgrem_path, im_path)

print ('!!! check consistency in lens: ',  len(bgrem_ims), 
       len(org_ims), len(src_ims))

print ('!!! check consistency in shapes: ', bgrem_ims[0].shape, org_ims[0].shape, src_ims[0].shape)

print ('check approx mean of counts: ', np.mean(bgrem_ims[10]), 
       np.mean(org_ims[10]), np.mean(src_ims[10]))

print ('check sum of counts: ', bgrem_ims[20].sum(), org_ims[20].sum(), src_ims[20].sum())


#################
# plot the distributions
##################

# Compute per-image means and sums
bgrem_means = bgrem_ims.mean(axis=(1, 2, 3)) # org shape (N, H, W, 1)
org_means = org_ims.mean(axis=(1, 2, 3))
src_means = src_ims.mean(axis=(1, 2, 3))

bgrem_sums = bgrem_ims.sum(axis=(1, 2, 3))
org_sums = org_ims.sum(axis=(1, 2, 3))
src_sums = src_ims.sum(axis=(1, 2, 3))

# plots
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# hist. of means
axes[0, 0].hist(bgrem_means, bins=50, alpha=0.7)
axes[0, 0].set_title("Mean counts: Bg-removed")
axes[0, 1].hist(org_means, bins=50, alpha=0.7)
axes[0, 1].set_title("Mean counts: Bg + Src.")
axes[0, 2].hist(src_means, bins=50, alpha=0.7)
axes[0, 2].set_title("Mean counts: Src. Only")

# hist. of sums
axes[1, 0].hist(np.log10(bgrem_sums), bins=50, alpha=0.7)
axes[1, 0].set_title("Log (Tot. counts): Bg-removed")
axes[1, 1].hist(np.log10(org_sums), bins=50, alpha=0.7)
axes[1, 1].set_title("Log (Tot. counts): Bg + Src.")
axes[1, 2].hist(np.log10(src_sums), bins=50, alpha=0.7)
axes[1, 2].set_title("Log (Tot. counts): Src. Only")

#### labels
axes[0, 0].set_ylabel('Counts')
axes[1, 0].set_ylabel('Counts')

for ax in axes.flat:
    ax.grid(True)
    
    ax.set_xlabel("Pixel Vals.")


plt.tight_layout()
plt.savefig('./bgrem_counts_dist-Mean-Sum.png', dpi=200)
plt.show()

##########
# scatter plot for means
# from hist. distrib., means are aggregated towards zero
##########
epsilon = 1e-2
fig = plt.figure(figsize=(8, 5))
plt.scatter(np.log10(src_means + epsilon), 
            np.log10(bgrem_means+epsilon), s=10, alpha=0.5, label='Bgrem')
plt.scatter(np.log10(src_means + epsilon), 
            np.log10(src_means + epsilon), s=10, alpha=0.4, color='red', label='True')

plt.xlabel("Log10 (True Source Ims Mean)")
plt.ylabel("Log10 (Bgrem Ims Mean)")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('./bgrem_mean_vs_src_mean.png', dpi=200)


####################
# let's just also add a residual on the same plot as subplot
####################

residuals = np.log10(bgrem_means + epsilon) - np.log10(src_means + epsilon)

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

# Top: scatter plot
axs[0].scatter(np.log10(src_means + epsilon), 
               np.log10(bgrem_means + epsilon), s=10, alpha=0.5, label='Image Mean (BGRem)')
axs[0].plot(np.log10(src_means + epsilon), 
            np.log10(src_means + epsilon), 'r--', alpha=0.7, label='Image Mean (Only Source)')

axs[0].set_ylabel("Log10 (BGRem Images: Pixel Mean)", fontsize=14)
axs[0].legend(fontsize=13)
axs[0].grid(True)

# Bottom: residuals
axs[1].scatter(np.log10(src_means + epsilon), residuals, s=10, alpha=0.4)
axs[1].axhline(0, color='red', linestyle='--', alpha=0.8)
axs[1].set_ylabel("Residual", fontsize=14)
axs[1].set_xlabel("Log10 (Only Source Images: Pixel Mean)", fontsize=14)
axs[1].grid(True)

plt.tight_layout()
plt.savefig('./bgrem_mean_vs_src_mean_with_residual.png', dpi=200)
plt.show()

####################
# let's check source extraction
####################
extract_sources(bgrem_ims[10, :, :, 0], im_name = im_names[10], save_path=home_path, type_im='bgrem')
extract_sources(org_ims[10, :, :, 0], im_name = im_names[10], save_path=home_path, type_im='iem_src')
extract_sources(src_ims[10, :, :, 0], im_name = im_names[10], save_path=home_path, type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[100, :, :, 0], im_name = im_names[100], save_path=home_path, 
                type_im='bgrem')
extract_sources(org_ims[100, :, :, 0], im_name = im_names[100], save_path=home_path, 
                type_im='iem_src')
extract_sources(src_ims[100, :, :, 0], im_name = im_names[100], save_path=home_path, 
                type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[200, :, :, 0], im_name = im_names[200], save_path=home_path, 
                type_im='bgrem')
extract_sources(org_ims[200, :, :, 0], im_name = im_names[200], save_path=home_path, 
                type_im='iem_src')
extract_sources(src_ims[200, :, :, 0], im_name = im_names[200], save_path=home_path, 
                type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[220, :, :, 0], im_name = im_names[220], save_path=home_path, type_im='bgrem')
extract_sources(org_ims[220, :, :, 0], im_name = im_names[220], save_path=home_path, type_im='iem_src')
extract_sources(src_ims[220, :, :, 0], im_name = im_names[220], save_path=home_path, type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[230, :, :, 0], im_name = im_names[230], save_path=home_path, 
                type_im='bgrem')
extract_sources(org_ims[230, :, :, 0], im_name = im_names[230], save_path=home_path, 
                type_im='iem_src')
extract_sources(src_ims[230, :, :, 0], im_name = im_names[230], save_path=home_path, 
                type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[240, :, :, 0], im_name = im_names[240], save_path=home_path, 
                type_im='bgrem')
extract_sources(org_ims[240, :, :, 0], im_name = im_names[240], save_path=home_path, 
                type_im='iem_src')
extract_sources(src_ims[240, :, :, 0], im_name = im_names[240], save_path=home_path, 
                type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

extract_sources(bgrem_ims[290, :, :, 0], im_name = im_names[290], save_path=home_path, 
                type_im='bgrem')
extract_sources(org_ims[290, :, :, 0], im_name = im_names[290], save_path=home_path, 
                type_im='iem_src')
extract_sources(src_ims[290, :, :, 0], im_name = im_names[290], save_path=home_path, 
                type_im='src')

print ("!!!!!!!!! Finished !!!!!!!!!")

###################
# define how to calculate tp and fp
###################


def is_true_positive(truth_objects, predicted_objects, i):
    distances = np.sqrt((truth_objects['x'] - predicted_objects['x'][i])**2 + 
                        (truth_objects['y'] - predicted_objects['y'][i])**2)
    return(np.any(distances<5))

def true_false_positives(truth_objects, predicted_objects):
    tp = []
    fp = []
    for i in range(len(predicted_objects)):
        if is_true_positive(truth_objects, predicted_objects, i):
            tp += [i]
        else:
            fp += [i]
    return(len(tp), len(fp))




N=100

bgfactor_groundtruth = 3
bgfactors_BGrem = np.linspace(1, 10, N)
bgfactors_SE = np.linspace(1, 10, N)
tp_BGRem = np.zeros(N)
fp_BGRem = np.zeros(N)
tp_SE = np.zeros(N)
fp_SE = np.zeros(N)

for i in range(len(src_ims)):
    bkg = sep.Background(src_ims[i,:,:,0])
    truth_objects = sep.extract(src_ims[i,:,:,0], bgfactor_groundtruth, err=bkg.globalrms)

    for j in range(N):
        bkg = sep.Background(bgrem_ims[i,:,:,0])
        predicted_objects_bgrem = sep.extract(bgrem_ims[i,:,:,0], 
                                        bgfactors_BGrem[j], err=bkg.globalrms)
        tp_BGRem_j, fp_BGRem_j = true_false_positives(truth_objects, predicted_objects_bgrem)
        tp_BGRem[j] += tp_BGRem_j
        fp_BGRem[j] += fp_BGRem_j

        bkg = sep.Background(org_ims[i,:,:,0])
        predicted_objects_org = sep.extract(org_ims[i,:,:,0], bgfactors_SE[j], err=bkg.globalrms)
        tp_SE_j, fp_SE_j = true_false_positives(truth_objects, predicted_objects_org)
        tp_SE[j] += tp_SE_j
        fp_SE[j] += fp_SE_j
    print(str(100*i//len(org_ims))+'%',end='\r')



plt.figure()
plt.plot(fp_SE, tp_SE, 'o', label='Standard SEP Bkg. Removal', alpha=0.7)
plt.plot(fp_BGRem, tp_BGRem, 'o', label='BGRem as pre-processing step', alpha=0.6)
#plt.plot(fp_BGRemSE,tp_BGRemSE,'o',label='SExtractor and BGRem background removal')
plt.xlabel('False positives')
plt.ylabel('True positives')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('Number of true sources found as a function of number of bogus sources found')
plt.legend(loc='lower right')
plt.savefig('./TP-FP-SEP_vs_BGREM.png', dpi=200)
#plt.xlim([-2,50])
# plt.show()

##########
# sep version ---> 1.4.1
##########