1先验知识获取的材料
论文，代码

2关于模式mode
这个项目代码中显示有好几种模式，分别是full，fullv2，two还有none 
其中除了训练架构参数的时候，其余都是使用的none模式
而在训练架构参数的时候，有几种可选模式
根据论文的表述来说，应该是使用的two，只有用two模式才挑选两个出来进行比较，实现了求梯度时的资源轻量化，然后进行权重的更新，然后进行总体权重的rescale，这样使得其它未参与本次迭代过程的原语的权重在softmax之后不变
但是default的训练架构参数的模式是fullv2，fullv2模式利用了论文附录处提到的另一种方式去求权重的梯度
在这个模式下，不能实现求梯度时的资源轻量化，不能实现N->2的资源减少，同时我也去试过这种模式，发现能塞的并不多(当然这是在我错误的代码实现之下的结果，因为从profiler已经知道我的代码有问题)

3关于求梯度的两种模式，论文中有两种求梯度的模式，一种是用强化学习，一种是通过文章中的公式4进行表示，强化学习的那种我没操作过

4总的来说有几个文件
一个是placeholder为单位下的一些操作，主要是对于权重的一些操作，代码在proxyless.py里面
还有就是挑选的代码在proxyless_trainer.py里面，这个是采用一边训原语的内部参数w，一边训架构参数arch的思路，经过一定时间的warmup之后才去训练arch(参照了原来的代码)