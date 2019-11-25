if __name__ == '__main__':
    def _generator():
        with open('train_jieba_cut_all.txt', 'r') as f:
            for line in f:
                print(line)

"""
浅拷贝是对一个对象父级的拷贝
可以分为两种情况：
拷贝对象可散列：处理逻辑就是直接引用目标对象的内存地址
拷贝对象非散列：处理逻辑就是分配新的内存空间存放。但是不包括内部元素

深拷贝就是对整个对象，包括容器对象的子对象，全部进行拷贝全部都会分配新的内存空间

垃圾回收机制：
主要使用引用计数来跟踪和回收对象，释放内存
在此基础之上，引入标记-清除算法，解决容器对象可能引起的循环引用问题
再通过分代回收，以空间换时间的方法来提升垃圾回收的效率
---引用计数---
每个对象都会维护一个自己的引用统计，有新的对象引用时，统计计数就+1，反之就-1，当引用计数为0的时候，该对象生命周期结束
优点就是简单，实时性强。缺点都还好，主要是有一个致命的循环引用。
---标记清除---
按需分配?没有空闲内存的时候从寄存器和程序栈上的引用处罚，遍历以对象为节点处罚，将所有可访问到的对象打上标记，然后清扫一遍内存空间，把未标记的对象释放????what
---分代回收---
将系统中的所有内存块根据其存活时间分为不同的三类集合。
三代中也存在垃圾，但是这些垃圾的回收会因为此机制而被延迟。
"""