import numpy as np
import timeit

class PrimalAfimEscala():
    def __init__(self, A, b, c, **options):
        # tolerance
        self._e = options.get('e', np.power(10.0, -3))

        # alpha parameter
        self._alpha = options.get('alpha', 0.9)
        self._alpha_k = self._alpha

        # maximum iterations parameter
        self._MAXITERATIONS = options.get('MAXITERATIONS', 100)

        # parameters for bigM method
        self._bigM = options.get('bigM', 1000.0)

        self._xk = options.get('x0', False)

        self._rk = 0.0
        self._wk = 0.0
        self._dk = 0.0

        self._sigma_c = 0.0
        self._sigma_d = 0.0
        self._sigma_p = 0.0

        # objective function value
        self._z = 0.0

        # number of iterations
        self._k = 1

        # lets store the output here
        self._output = {
            "finalResult": "",
            "output": []
        }

        # limit the precision printing to 3 decimals
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)

        if not (self._xk):
            # lets store the original A,b,c set
            self._originalAbc = {'A': A, 'b': b, 'c': c}

            self._A = np.insert(A, A.shape[1], b - np.dot(A, np.ones(A.shape[1])), axis=1)
            self._b = b
            self._c = np.insert(c, c.shape[0], self._bigM, axis=0)
            self._xk = np.ones(c.shape[0] + 1)
        else:
            self._A = A
            self._b = b
            self._c = c

        #self._Xk = np.diag(self._xk)

    def _setStartTime(self):
        self._starttime = timeit.default_timer()

    def _setEndTime(self):
        self._endtime = timeit.default_timer()

    def _setConsumedTime(self):
        self._consumedtime = self._endtime - self._starttime
        if (self._output):
            self._output['consumedtime'] = self._consumedtime


    def _calcz(self):
        self._z = np.array([np.dot(self._c, self._xk)])

    def _calcXk(self):
        self._Xk = np.diag(self._xk)

    def _calcwk(self):
        a = np.dot(
            np.dot(self._A, np.power(self._Xk, 2)),
            np.transpose(self._A)
        )

        b = np.dot(
            np.dot(self._A, np.power(self._Xk, 2)),
            self._c
        )

        self._wk = np.dot(np.linalg.inv(a), b)
        # self._wk = np.linalg.solve(a, b)

    def _calcrk(self):
        self._rk = self._c - np.dot(np.transpose(self._A), self._wk)

    def _calc_sigma_p(self):
        self._sigma_p = np.linalg.norm(np.dot(self._A, self._xk) - self._b)
        self._sigma_p = self._sigma_p / (np.linalg.norm(self._b) + 1)

    def _calc_sigma_d(self):
        self._sigma_d = np.linalg.norm(self._rk[self._rk < 0])
        self._sigma_d = self._sigma_d / (np.linalg.norm(self._c[self._rk < 0]) + 1)
        # self._sigma_d = self._sigma_d/(np.linalg.norm(self._c[self._c < 0]) + 1)

    def _calc_sigma_c(self):
        self._sigma_c = np.dot(np.transpose(self._c), self._xk) - np.dot(np.transpose(self._b), self._wk)

    def _calc_dk(self):
        self._dk = -np.dot(self._Xk, self._rk)

    def _calc_alpha_k(self):
        self._alpha_k = self._alpha * np.amin(1 / (-self._dk[self._dk < 0]))

    def _isUnbounded(self):
        if (np.all(np.greater_equal(self._dk, 0))):
            self._setFinalResult("Is Unbounded.")
            return True

        return False

    def _updatexk(self):
        self._xk_prev = self._xk
        self._xk = self._xk + self._alpha_k * np.dot(self._Xk, self._dk)

    def _updateIteration(self):
        self._k = self._k + 1

    def _isOptimal(self):
        if np.less_equal(self._sigma_d, self._e) and np.less_equal(self._sigma_c, self._e):
            self._setFinalResult("Is Optimal!")
            return True

        return False

    def _checkMaxIterations(self):
        if (self._k > self._MAXITERATIONS):
            self._setFinalResult("Maximum Iterations reached!")
            return True
        return False

    def _setFinalResult(self, result):
        self._output["finalResult"] = result

    def _storeOutput(self):
        self._output['output'].append({
            'k': self._k,
            'xk': self._xk,
            'z': self._z,
            'alpha_k': self._alpha_k,
            'rk': self._rk,
            'wk': self._wk,
            'dk': self._dk,
            'sigma_d': self._sigma_d,
            'sigma_c': self._sigma_c,
        })
        pass

    # ////////
    # PUBLIC METHODS
    # ////////
    def getFancyOutput(self):
        output = ""
        output += self._output['finalResult'] + "\n"
        output += "k,z,sigma_d,sigma_c,wk,xk" + "\n"
        for line in self._output['output']:
            output += str(line['k']) + "|"
            output += str(line['z']) + "|"
            output += str(line['sigma_d']) + "|"
            output += str(line['sigma_c']) + "|"
            output += str(line['wk']) + "|"
            output += str(line['xk']) + "\n"

        return output

    def getOutput(self):
        return self._output

    def run(self):

        self._setStartTime()

        while 1:
            self._storeOutput()

            self._calcz()

            self._calcXk()
            self._calcwk()
            self._calcrk()

            self._calc_sigma_p()
            self._calc_sigma_d()
            self._calc_sigma_c()

            if (self._isOptimal()): break

            self._calc_dk()

            if self._isUnbounded(): break

            self._calc_alpha_k()

            self._updatexk()

            self._updateIteration()

            if (self._checkMaxIterations()): break

        self._setEndTime()
        self._setConsumedTime()

class DualAfimEscala():
    def __init__(self, A, b, c, **options):

        # tolerance
        self._e = options.get('e', np.power(10.0, -8))

        # alpha parameter
        self._alpha = options.get('alpha', 0.9)

        # maximum iterations parameter
        self._MAXITERATIONS = options.get('MAXITERATIONS', 100)

        # parameters for bigM method
        self._theta = options.get('theta', 2)
        self._bigM = options.get('bigM', 1000.0)

        # final variables 
        self._sk = np.zeros_like(A.shape[0])
        self._wk = np.zeros_like(b)

        # previous artifical variable value
        # needs to be check because of the bigM
        self._wa_prev = 0.0
        self._xk = np.zeros_like(c)

        # directions
        self._dw = 0.0
        self._ds = 0.0

        # objective function value
        self._z = 0.0

        # iterations counter
        self._k = 0

        # complementary slackness
        self._sigma_c = 0.0

        # lets store the output here
        self._output = {
            "finalResult": "",
            "output": []
        }

        # ===========================================
        # numpy parameters
        # limit the precision printing to 3 decimals
        # ===========================================
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        if (np.all(c > 0)):
            # dont need to use the bigM
            self._check_wa = False

            self._A = A
            self._b = b
            self._c = c

            self._wk = options.get('w0', False)
            self._sk = options.get('s0', False)

            if not (self._wk):
                self._wk = np.zeros(A.shape[0])

            if not (self._sk):
                self._sk = c

        else:
            # prepare for the bigM method
            self._check_wa = True

            # lets store the original A,b,c set
            self._originalAbc = {'A': A, 'b': b, 'c': c}

            # prepare the "p" vector
            self._p = np.zeros(c.shape)
            self._p[c <= 0] = 1.0

            # prepare the c_bar parameter
            self._c_bar = np.max(np.absolute(c))

            # the start s is simply a math with the c vector
            self._sk = c + self._theta * self._c_bar * self._p

            # the start w will have the same size as b, but an artificial variable at the end
            self._wk = np.zeros(b.shape[0] + 1)
            self._wk[-1] = -self._theta * self._c_bar

            # add the "p" vector as a line at the bottom of A. We'll use Transpose[A]
            # along the algorithm
            self._A = np.insert(A, A.shape[0], self._p, axis=0)

            # the c vector is the same
            self._c = c

            # add the bigM value at the end of b vector. We'll it as a cost vector (Transpose[b])
            self._b = np.insert(b, b.shape[0], [self._bigM], axis=0)

    def _setStartTime(self):
        self._starttime = timeit.default_timer()

    def _setEndTime(self):
        self._endtime = timeit.default_timer()

    def _setConsumedTime(self):
        self._consumedtime = self._endtime - self._starttime
        if (self._output):
            self._output['consumedtime'] = self._consumedtime

    def _calcz(self):
        #self._z = np.array([np.dot(np.transpose(self._b), self._wk)])
        self._z = np.array([np.dot(self._b, self._wk)])

    def _calcSK(self):
        self._SK = np.diag(self._sk)

    def _calcdw(self):
        self._dw = np.linalg.solve(
            np.dot(
                self._A,
                np.dot(
                    np.linalg.inv(np.power(self._SK, 2)),
                    np.transpose(self._A)
                )
            ),
            self._b
        )

    def _calcds(self):
        self._ds = -np.dot(np.transpose(self._A), self._dw)

    def _calcxk(self):
        self._xk = -np.dot(np.linalg.inv(np.power(self._SK, 2)), self._ds)

    def _calc_sigma_c(self):
        self._sigma_c = np.dot(np.transpose(self._c), self._xk) - np.dot(np.transpose(self._b), self._wk)

    def _calc_beta_k(self):
        self._beta_k = self._alpha * np.amin(self._sk[self._ds < 0] / -self._ds[self._ds < 0])

    def _updatewk(self):
        if (self._check_wa):
            self._wa_prev = self._wk[-1]

        self._wk = self._wk + self._beta_k * self._dw

    def _updatesk(self):
        self._sk = self._sk + self._beta_k * self._ds

    def _updateIteration(self):
        self._k = self._k + 1

    def _checkMaxIterations(self):
        if (self._k > self._MAXITERATIONS):
            self._setFinalResult('Max Iterations Reached')
            return True
        return False

    def _isUnbounded(self):
        if (np.all(np.equal(self._ds, 0)) or np.all(np.greater(self._ds, 0))):
            self._setFinalResult('Is Unbounded!')
            return True
        return False

    def _isOptimal(self):
        if (np.all(np.greater_equal(np.round(self._xk,4), 0)) and np.all(np.less_equal(self._sigma_c, self._e))):
            self._setFinalResult('Is Optimal!')
            return True
        return False

    def _check_wa_crossed_zero(self):

        wa = self._wk[-1]

        if (wa > 0 and self._wa_prev < 0):
            self._setFinalResult('wa crossed zero or is very near!')
            return True
        return False

    def _restore_original_problem(self):

        self._check_wa = False

        self._A = self._originalAbc.get("A")
        self._b = self._originalAbc.get("b")
        self._c = self._originalAbc.get("c")

        self._sk = self._sk + self._p * self._wk[-1]

        #remove last element
        self._wk = self._wk[:-1]

        self._k = 0
        self._setFinalResult("Restarting the problem...")
        pass

    def _setFinalResult(self, result):
        self._output["finalResult"] += " " + result

    def _storeOutput(self):
        self._output['output'].append({
            'k': self._k,
            'z': self._z,
            'xk': self._xk,
            'wk': self._wk,
            'dw': self._dw,
            'ds': self._ds,
            'sigma_c': self._sigma_c,
        })

    # ////////
    # PUBLIC METHODS
    # ////////
    # k,z,sigma_d,sigma_c,wk,xk
    def getFancyOutput(self):
        output = ""
        output += self._output['finalResult'] + "\n"
        output += "k,z,sigma_c,wk,xk" + "\n"
        for line in self._output['output']:
            output += str(line['k']) + "|"
            output += str(line['z']) + "|"
            output += str(line['sigma_c']) + "|"
            output += str(line['wk']) + "|"
            output += str(line['xk']) + "\n"

        return output

    def getOutput(self):
        return self._output

    def run(self):

        self._setStartTime()
        self._storeOutput()

        while 1:

            self._calcz()

            if (self._check_wa):
                if (self._check_wa_crossed_zero()):
                    # restore the original problem
                    self._restore_original_problem()
                    continue

            self._calcSK()
            self._calcdw()
            self._calcds()

            if self._isUnbounded():
                self._storeOutput()
                break

            self._calcxk()
            self._calc_sigma_c()

            if (self._isOptimal()): break

            self._calc_beta_k()
            self._updatewk()
            self._updatesk()
            self._updateIteration()

            if (self._checkMaxIterations()): break

            self._storeOutput()

        self._setEndTime()
        self._setConsumedTime()

class PrimalDualAfimEscala():
    def __init__(self, A, b, c, **options):

        # tolerances
        self._e1 = options.get('e1', np.power(10.0, -8))

        # alpha parameter
        self._alpha = options.get('alpha', 0.9)

        # sigma u parameter
        self._sigma_u = options.get('sigma_u', 0.9)

        # the problem itself
        self._A = A
        self._b = b
        self._c = c

        #iterations counter
        self._k = 1

        # initial solutions
        self._xk = options.get('x0', np.ones_like(self._c))
        self._wk = options.get('w0', np.ones_like(self._b))
        self._sk = options.get('s0', np.ones(self._A.shape[1]))
        self._e = np.ones_like(self._c)

        self._MAXITERATIONS = options.get('MAXITERATIONS', 50)

        # lets store the output here
        self._output = {
            "finalResult": "",
            "output": []
        }

    def _setFinalResult(self, result):
        self._output["finalResult"] += " " + result

    def _storeOutput(self):
        self._output['output'].append({
            'k': self._k,
            'z': self._z,
            'xk': self._xk,
            'wk': self._wk,
            'dw': self._dw,
            'ds': self._ds
        })

    def _setStartTime(self):
        self._starttime = timeit.default_timer()

    def _setEndTime(self):
        self._endtime = timeit.default_timer()

    def _setConsumedTime(self):
        self._consumedtime = self._endtime - self._starttime
        if (self._output):
            self._output['consumedtime'] = self._consumedtime

    def _calcz(self):
        self._z = np.array([np.dot(self._c, self._xk)])

    def _calcXk(self):
        self._Xk = np.diag(self._xk)

    def _calcSk(self):
        self._Sk = np.diag(self._sk)

    def _calc_mi_k(self):
        self._mi_k = self._sigma_u*(np.dot(np.transpose(self._xk),self._sk))
        self._mi_k = self._mi_k/(self._xk.shape[0])

    def _calc_Dk(self):
        self._Dk = np.dot(self._Xk,np.linalg.inv(self._Sk))
        self._Dk = np.sqrt(self._Dk)

    def _calc_Pk(self):
        self._Pk = np.dot(self._Dk,np.transpose(self._A))
        self._Pk = np.dot(self._Pk, np.linalg.inv(np.dot(np.dot(self._A,np.power(self._Dk,2)),np.transpose(self._A))))
        self._Pk = np.dot(self._Pk,np.dot(self._A,self._Dk))

    def _calc_vk(self):
        self._vk = np.dot(np.linalg.inv(self._Xk),self._Dk)
        self._vk = np.dot(self._vk, np.dot(self._mi_k,self._e) - np.dot(np.dot(self._Xk,self._Sk),self._e) )

    def _calc_dx(self):
        self._dx = np.dot(self._Dk,(np.ones_like(self._Pk) - self._Pk))
        self._dx = np.dot(self._dx,self._vk)

    def _calc_dw(self):
        self._dw = -np.linalg.inv(np.dot(np.dot(self._A,np.power(self._Dk,2)),np.transpose(self._A)))
        self._dw = np.dot(self._dw, np.dot(np.dot(self._A,self._Dk),self._vk))

    def _calc_ds(self):
        self._ds = np.dot(np.linalg.inv(self._Dk),self._Pk)
        self._ds = np.dot(self._ds,self._vk)

    def _calc_beta_p(self):
        #self._beta_p = np.max(np.append([1.0],-(self._dx  /(self._alpha*self._xk))))
        self._beta_p = np.min(np.append([1.0],-(self._xk[self._dx < 0] /self._dx[self._dx < 0])))


    def _calc_beta_d(self):
        #self._beta_d = np.max(np.append([1.0],-(self._ds  /(self._alpha*self._sk))))
        self._beta_d = np.min(np.append([1.0],-(self._sk[self._ds < 0] /self._ds[self._ds < 0])))

    def _update_xk(self):
        self._xk = self._xk + self._beta_p*self._dx

    def _update_wk(self):
        self._wk = self._wk + self._beta_d*self._dw

    def _update_sk(self):
        self._sk = self._sk + self._beta_d*self._ds

    def _updateIteration(self):
        self._k += 1

    def _checkMaxIterations(self):
        if (self._k > self._MAXITERATIONS):
            self._setFinalResult('Max Iterations Reached')
            return True
        return False

    def _isOptimal(self):
        if (np.all(np.less(np.dot(self._sk,self._xk),self._e1))):
            return True
        return False

    def getFancyOutput(self):
        output = ""
        output += self._output['finalResult'] + "\n"
        output += "k,z,sigma_c,wk,xk" + "\n"
        for line in self._output['output']:
            output += str(line['k']) + "|"
            output += str(line['z']) + "|"
            output += str(line['wk']) + "|"
            output += str(line['xk']) + "\n"

        return output

    def run(self):



        self._setStartTime()
        while 1:

            if (self._isOptimal()): break

            self._calcz()
            self._calcSk()
            self._calcXk()

            self._calc_mi_k()

            self._calc_Dk()
            self._calc_Pk()
            self._calc_vk()

            self._calc_dx()
            self._calc_dw()
            self._calc_ds()

            self._calc_beta_d()
            self._calc_beta_p()

            self._update_wk()
            self._update_sk()
            self._update_xk()

            self._updateIteration()

            self._storeOutput()

            if (self._checkMaxIterations()): break


        self._setEndTime()
        self._setConsumedTime()