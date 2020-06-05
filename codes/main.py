from deepDream import *
import os
'''
dream all 1000 labels, multiple copies of a label by changing learning rate and  gaussian kernel
'''
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

output_dir = 'output/vgg19_cifar_conv16/'
def main():
    nItrs = [1000]
    # lrs = [0.08,0.1,0.12,0.14]
    lrs = [0.14]
    # sigmas = [0.4.png,0.42,0.44,0.46,0.48,0.5]
    sigmas = [0.4]
    image_dict = {}

    model = DeepDream()
    for sigma in sigmas:
        model.createGaussianFilter(sigma=sigma) # create gaussian filter
        for label in range(0, 512):
            for lr in lrs:
                for nItr in nItrs:
                    outputImage = model.dream(label=label,nItr=nItr,lr=lr) # dream
                    fileName = "dream_"+str(label)+"_"+str(nItr)+"_"+str(lr)+"_"+str(sigma)+".png"
                    print (fileName)
                    image_dict[fileName]=outputImage
                    # print(params)
            if (label % 8 == 7): # clear out by saving the output images
                for name,image in image_dict.items():
                    model.save(image, output_dir+name) # save the images

                image_dict.clear() # clear the dictionary

def predict(img):
    model = DeepDream()
    res = list(model.predict(img)[0])
    print(res)
    return np.argmax(res)


def label_filters(img, label):
    model = DeepDream()
    res = model.label_img(img, label)
    return res


def batch_label():
    files = os.listdir('output/vgg16_cifar10_fc3/')
    filter_pres = {}
    for f in files:
        filter_pres[f] = {}
        for label in range(10):
            filter_pres[f][label] = label_filters('output/vgg16_cifar10_fc3/'+f, label)[0, label, 0, 0]
        filter_pres[f]['pre'] = max(filter_pres[f], key=lambda i:filter_pres[f][i])
    for k, v in filter_pres.items():
        print(k, v)

if __name__ == "__main__":
    main()


