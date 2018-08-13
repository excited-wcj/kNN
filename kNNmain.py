#第一部分：生成kNN分类器
import kNN
from numpy import *
group, labels = kNN.createDataSet()
group
labels
kNN.classify0([0,0], group, labels, 3)
#第二部分：示例
import kNN
from numpy import *
import importlib as imp
#Q 这行已经在kNN中声明过了，还要在写一遍。同一层封装下模块(.py)的函数需要调用库或者脚本而非全局使用
imp.reload(kNN)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
#需要reload(kNN) ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
 #Q 这两张图片怎么显示轴便签？而且还是自定义标签？要去学习malplotlib的开发文档
plt.show()  
#第三部分：
import kNN
imp.reload(kNN) #import importlib as imp
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
normMat
ranges
minVals
#第四部分：对陌生异性好感度预测结果
kNN.classifyPerson()
#第五部分：手写识别系统
#Q 测试样本与训练样本的实质区别
testVector = kNN.img2vector('testDigits/0_13.txt')
#Q 识别文件名称分别有什么方法识别带文件格式和不带文件格式的两种？
testVector[0, 0:31]
testVector[0, 32:63]
kNN.handwritingClassTest()
#Q 提高准确率的（即降低错误率的方式）：改变k值，随机选取训练样本，改变训练样本的熟
