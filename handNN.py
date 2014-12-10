from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.plotting      import MultilinePlotter

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

import csv

def runNN():
    # data set CSV
    _trncsv = '../ml_pybrain/case1_send1/inc_attr_30_train.csv'
    _tstcsv = '../ml_pybrain/case1_send1/inc_attr_30_test.csv'
    #_attrnum = 30 # attrnum is imput dim, get later auto
    _classnum = 2
    # num of hidden layer
    _hidden = 12
    # max epochs
    _maxepochs = 100
    # learn val
    _learningrate = 0.013 # default 0.01
    _momentum = 0.03 # default 0.0, tutorial 0.1
    _lrdecay = 1.0 # default 1.0
    _weightdecay = 0.01 # default 0.0, tutorial 0.01
    # save training log path
    _logpath = 'att30class2_2.log'
    # graph
    _graphymax = 15

    '''#build 3 class
    means = [(-1,0),(2,4),(3,1)]
    cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    alldata = ClassificationDataSet(2, 1, nb_classes=3)
    for n in xrange(400):
        for klass in range(3):
            input = multivariate_normal(means[klass],cov[klass])
            alldata.addSample(input, [klass])

    #split 25% test, 75% train
    tstdata, trndata = alldata.splitWithProportion(0.25)


    ''' #read csv
    # read train data
    dictdata = csv.DictReader(open(_trncsv, 'r'))
    data  = [[row[f] for f in dictdata.fieldnames] for row in dictdata]
    # from 0th to last-1 col are training data set
    train = [[float(elm) for elm in row[0:-1]] for row in data]
    # last col is target data set, convert from ONE start to ZERO start
    target = [[int(row[-1])-1] for row in data]
    # get input dim
    _attrnum = len(train[0])
    # set DataSet
    trndata = ClassificationDataSet(_attrnum, 1, nb_classes=_classnum)
    trndata.setField('input', train)
    trndata.setField('target', target)

    # read test data
    dictdata = None
    dictdata = csv.DictReader(open(_tstcsv, 'r'))
    data  = [[row[f] for f in dictdata.fieldnames] for row in dictdata]
    # from 0th to last-1 col are training data set
    train = [[float(elm) for elm in row[0:-1]] for row in data]
    # last col is target data set, convert from ONE start to ZERO start
    target = [[int(row[-1])-1] for row in data]
    # set DataSet
    tstdata = ClassificationDataSet(_attrnum, 1, nb_classes=_classnum)
    tstdata.setField('input', train)
    tstdata.setField('target', target)

    #'''

    # 1-of-k representation
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]

    # build network and tariner
    fnn = buildNetwork( trndata.indim, _hidden, trndata.outdim, outclass=SoftmaxLayer)
    #trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
    trainer = BackpropTrainer(fnn, dataset=trndata, verbose=True,
        learningrate = _learningrate,
        momentum = _momentum,
        lrdecay = _lrdecay,
        weightdecay = _weightdecay
        )

    # setup graph
    xmax = _maxepochs
    ymax = _graphymax
    figure(figsize=[12,8])
    ion()
    draw()
    graph = MultilinePlotter(xlim=[1, xmax], ylim=[0, ymax])
    graph.setLineStyle([0,1], linewidth=2)
    graph.setLabels(x='epoch', y='error %')
    graph.setLegend(['training', 'test'], loc='upper right')
    graph.update()
    draw()

    # setup storage training curve
    trainx = []
    trny = []
    tsty = []

    # start training
    for i in range(_maxepochs):
        # train
        trainer.trainEpochs(1)
        # test by train/test
        trnresult = percentError(trainer.testOnClassData(), trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        print "epoch: %4d" % trainer.totalepochs, \
            "  train error: %5.2f%%" % trnresult, \
            "  test error: %5.2f%%" % tstresult
        # store curve
        trainx.append(i+1)
        trny.append(trnresult)
        tsty.append(tstresult)

        # draw graph
        graph.addData(0, i+1, trnresult)
        graph.addData(1, i+1, tstresult)
        graph.update()
        draw()

    # save log
    f = csv.writer(open(_logpath, 'w'))
    # data prop
    f.writerow(['train data num', len(trndata)])
    f.writerow(['test data num', len(tstdata)])
    f.writerow(['in / out dim', trndata.indim, trndata.outdim])
    # config
    f.writerow(['hidden', _hidden])
    f.writerow(['maxepochs', _maxepochs])
    f.writerow(['learningrate', _learningrate])
    f.writerow(['momentum', _momentum])
    f.writerow(['lrdecay', _lrdecay])
    f.writerow(['weightdecay', _weightdecay])
    # curve
    f.writerow(['epoch', 'train_err', 'test_err'])
    f.writerows([[trainx[r], trny[r], tsty[r]] for r in range(len(trainx))])
    #f.close()
