import numpy as np

from AfimEscala import *

data = [
    {
        'problem_name' : 'frannie',
        'A' : np.array([
            [0.5, 1.0, 1.0],
        ]),
        'b' : np.array([3.0]),
        'c' : np.array([-90.0, -150.0, 0.0]),
        'precisions' : [np.power(10.0, -3)],
        'alphas' : [0.95],
        'bigMs' : [1000.0],
        'algorithms' : ["PAE","DAE","PDAE"]
    },
    {
        'problem_name' : 'lista_exercicio_manual',
        'A' : np.array([
            [1.0, 1.0]
        ]),
        'b' : np.array([1.0]),
        'c' : np.array([2.0, 1.0]),
        'precisions' : [np.power(10.0, -3)],
        'alphas' : [0.95],
        'bigMs' : [1000.0],
        'algorithms' : ["PAE","DAE","PDAE"]

    },
    {
        'problem_name' : 'sapateiro',
        'A' : np.array([
            [2.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 1.0]
        ]),
        'b' : np.array([8.0,7.0,3.0]),
        'c' : np.array([-1.0, -1.0, 0.0 , 0.0 , 0.0]),
        'precisions' : [np.power(10.0, -3)],
        'alphas' : [0.95],
        'bigMs' : [1000.0],
        'algorithms' : ["PAE","DAE","PDAE"]

    },
    {
        'problem_name' : 'problem_computational_list_2',
        'A' : np.array([
            [1.0, 0.0, 0.0, 0.25, -8.0  , -1.0 , 9.0],
            [0.0, 1.0, 0.0, 0.50, -12.0 , -0.5 , 3.0],
            [0.0, 0.0, 1.0, 0.00,  0.0  ,  1.0 , 0.0],
        ]),
        'b' : np.array([0.0,0.0,1.0]),
        'c' : np.array([0.0,0.0,0.0,-0.75,20.0,-0.5,6.0]),
        'precisions' : [np.power(10.0, -3)],
        'alphas' : [0.95],
        'bigMs' : [1000.0],
        'algorithms' : ["PAE","DAE","PDAE"]
    },
    {
        'problem_name' : 'problem_computational_list_3',
        'A' : np.array([
            [1.0, 1.0, -1.0,  0.0, 0.0  ,  0.0 , 0.0],
            [1.0, 0.0,  0.0, -1.0, 0.0  ,  0.0 , 0.0],
            [1.0, 0.0,  0.0,  0.0, 1.0  ,  0.0 , 0.0],
            [0.0, 1.0,  0.0,  0.0, 0.0  , -1.0 , 0.0],
            [0.0, 1.0,  0.0,  0.0, 0.0  ,  0.0 , 1.0]
        ]),
        'b' : np.array([3.0,1.0,4.0,1.0,4.0]),
        'c' : np.array([10.0,20.0,0.0,0.0,0.0,0.0,0.0]),
        'precisions' : [np.power(10.0, -3),np.power(10.0, -5)],
        'alphas' : [0.915,0.955,0.995],
        'bigMs' : [1000.0],
        'algorithms' : ["PAE","DAE","PDAE"]
    }
]



#for d in data:
for d in [data[0]]:
    for algorithm in d['algorithms']:
        for e in d['precisions']:
            for alpha in d['alphas']:
                for bigM in d['bigMs']:

                    if algorithm == "PAE":
                        #continue
                        algo = PrimalAfimEscala(d['A'], d['b'], d['c'], e=e, alpha=alpha)
                    elif algorithm == "DAE":
                        #continue
                        algo = DualAfimEscala(d['A'], d['b'], d['c'], e=e, alpha=alpha)
                    elif algorithm == "PDAE":
                        algo = PrimalDualAfimEscala(d['A'], d['b'], d['c'], e=e, alpha=alpha)


                    algo.run()
                    print "Problem Name: ", d['problem_name']
                    print "Algorithm: ", algorithm
                    print "Precision: ", e
                    print "Alpha: ", alpha
                    print "Output: ", algo.getFancyOutput()
                    print "*****************************"
