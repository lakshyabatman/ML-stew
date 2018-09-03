import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
# parameters of third eye
malignant_mean1 = 37
malignant_stddev1 = 1
benign_mean1 = 32
benign_stddev_1 = 4

# parameters of comp
malignant_mean2 = 37
malignant_stddev2 = 2
benign_mean2 = 32
benign_stddev2 = 3

sample_size = 1000
# generate samples
malignant1 = np.random.normal(malignant_mean1,malignant_stddev1,(sample_size,))
malignant2 = np.random.normal(malignant_mean2,malignant_stddev2,(sample_size,))

benign1 = np.random.normal(benign_mean1,benign_stddev_1,(sample_size,))
benign2 = np.random.normal(benign_mean2,benign_stddev2,(sample_size,))

# find approximate PDF
n=sample_size
p1, xm1 = np.histogram(malignant1, bins=np.int(n/10)) # bin it into n = N//10 bins
# print(xm1)
xm1 = xm1[:-1] + (xm1[1] - xm1[0])/2   # convert bin edges to centers
mal_pdf1 = UnivariateSpline(xm1, p1, s=n)

p2, xb1 = np.histogram(benign1, bins=np.int(n/10)) # bin it into n = N//10 bins
xb1 = xb1[:-1] + (xb1[1] - xb1[0])/2   # convert bin edges to centers
ben_pdf1 = UnivariateSpline(xb1, p2, s=n)


p3, xm2 = np.histogram(malignant2, np.int(n/10)) # bin it into n = N//10 bins
xm2 = xm2[:-1] + (xm2[1] - xm2[0])/2   # convert bin edges to centers
mal_pdf2 = UnivariateSpline(xm2, p3, s=n)

p4, xb2 = np.histogram(benign2, np.int(n/10)) # bin it into n = N//10 bins
xb2 = xb2[:-1] + (xb2[1] - xb2[0])/2   # convert bin edges to centers
ben_pdf2 = UnivariateSpline(xb2, p4, s=n)

# find out confusion matrix entries
tp1 = np.zeros((100,))
fp1 = np.zeros((100,))
fn1 = np.zeros((100,))
tn1 = np.zeros((100,))

tp2 = np.zeros((100,))
fp2 = np.zeros((100,))
fn2 = np.zeros((100,))
tn2 = np.zeros((100,))
for i in range(100):
	threshold = i-1
	for k in range(sample_size):
		if(malignant1[k]>threshold):
			tp1[i]+=1
		else:
			fn1[i]+=1
		if(benign1[k]>threshold):
			fp1[i]+=1
		else:
			tn1[i]+=1
		if(malignant2[k]>threshold):
			tp2[i]+=1
		else:
			fn2[i]+=1
		if(benign2[k]>threshold):
			fp2[i]+=1
		else:
			tn2[i]+=1

tpr1 = tp1/(tp1+fn1)
fpr1 = fp1/(fp1+tn1)
auc1 = np.trapz(tpr1[::-1],fpr1[::-1])
print('AUC for Third Eye technologies is ',auc1)

tpr2 = tp2/(tp2+fn2)
fpr2 = fp2/(fp2+tn2)
auc2 = np.trapz(tpr2[::-1],fpr2[::-1])
print('AUC for competitor is ',auc2)
# 
plt.figure(1)
plt.plot(xm1, mal_pdf1(xm1),'r',label='Third eye - Malignant')
plt.plot(xb1, ben_pdf1(xb1),'g',label='Third eye - Benign')
plt.plot(xm2, mal_pdf2(xm2),'r--',label='Competitor - Malignant')
plt.plot(xb2, ben_pdf2(xb2),'g--',label='Competitor - Benign')

plt.title('Distribution of classes')
plt.ylabel('No of samples')
plt.xlabel('Temperature of tumour')
plt.legend()

plt.figure(2)
plt.plot(fpr1, tpr1,'b--',label='Third Eye')
plt.plot(fpr2, tpr2,'r--',label='Competitor')
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()