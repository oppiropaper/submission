from neural_tangents import stax
from collections import namedtuple
import neural_tangents as nt
import numpy as np
import jax
import jax.numpy as jnp
from utils import gen_rand_data_on_sphere, add_Gaussian_noise, quadratic_form
import math
from utils import get_sup

Kernel = namedtuple('Kernel', ['ntk', 'nngp'])

class DenseNN:
    '''
    Class to handle dense NN, corresponding ntk using neural tangents module. 
    Key attributes after initialization are init_fn, apply_fn, kernel_fn.
    init_fn is a funcion that initializaes the nn with random weights. 
    apply_fn does a forward pass through the network (after init_fn has been called).
    kernel_fn calculates the infinite width kernels associated to the network.
    For use see the corresponding wrappers below and for params see constructor below. 
    '''
    @staticmethod
    def activation_relu(s):
        def fn(x: float):
            res = jnp.maximum(x,0)
            return res**s
        
        def nngp_fn(cov12: float, x1: float, x2: float):
            return relu_pow_s(s, cov12, x1, x2)
            

        def d_nngp_fn(cov12, x1, x2):
            return s*relu_pow_s(s-1, cov12, x1, x2)

        return stax.Elementwise(fn, nngp_fn, d_nngp_fn)

    @staticmethod
    def activation_relu_numerical(s = 1, deg_for_numerical_approx = 25):
        '''
        static method to generate approximation to Relu^s.
        s - the power. 
        deg_for_numerical_approx: parameter for number of points to approximate the activation. Relevant only when 
                activation is not Relu but an approximation (see above). If too low, the predictions can return nans
                when posteriors are calculated on large enough number of points.
        '''
        def func(x):
            return jnp.maximum(x, 0) ** s
        return stax.ElementwiseNumerical(func, deg = deg_for_numerical_approx)

    def __init__(self, output_dim:int, hidden_layers_array, activation = None, W_std = 1.5, b_std = 0.05, parametrization = 'ntk'):
        '''
        Input:
        output_dim: dimension of the output.
        hidden_layers_array: array like of ints where ith int is the number of neurons in layer i.
        activation: activation function to be used. Use None for Relu and DenseNN.activation_relu(s, deg_for_numerical_approximation) for relu**s approximation.
        W_std: std of linear weights at initialization (does not affect ntk).
        b_std: std of bias at initialization (does not affect ntk)
        parametrization: the way to initialize. Affects only the nn and not the associated ntk (see neural tangents doc for more info).   
        '''
        self._params = None
        assert(len(hidden_layers_array) >= 1)
        layers = []
        if activation is None:
            activation_func = stax.Relu()
        else:
            activation_func = activation
        for num_of_units in hidden_layers_array:
            layers += [stax.Dense(num_of_units, W_std, b_std, parametrization), activation_func]
        layers += [stax.Dense(output_dim, W_std, b_std, parametrization)]
        self._init_fn, self._apply_fn, self._kernel_fn = stax.serial(*layers)


    def init_fn(self, input_shape, key = None):
        '''
        Wraps init_fn of a jax/neural tangents neural network object. 
        Inputs: 
        input_shape: int. Dimension of inputs of neural network. 
        '''
        if key is None:
            seed = np.uint32(int.from_bytes(np.random.bytes(4), 'big'))
            key = jax.random.PRNGKey(seed)
        _, self._params = self._init_fn(key, (-1, input_shape))


    def apply_fn(self, input, params = None):
        '''
        Wraps apply_fn of a jax/neural tangents neural network object.
        Inputs: 
            input: array or array like of dimension given in init_fn.
        '''
        assert(self._params is not None)
        params = self._params
        return self._apply_fn(params, input)    


    def kernel_fn(self, input1,  input2 = None, mode = 'ntk'):
        '''
        Wraps kernel_fn of a neural tangents neural network object.
        
        Inputs: 
        input: array or array like of shape (batch_size, *) where * can be any dimension! 
        (the kernel is calculated dependant on the dimension of input). 
        '''
        if input2 is None:
            res = self._kernel_fn(input1, get = mode)
        else:
            res = self._kernel_fn(input1, input2, mode)
        return res


    def generate_gp_samples(self, input, num_of_samples = 1, mode = 'ntk', mean = None):
        '''
        Generates values of a Gaussian Process at the points provided in input (the GP has 
        distribution specified by the kernel_fn).
        
        Input:
        input:  array or array like of shape (batch, *). These are the points at which the values of 
                the GP will be generated.
        num_of_samples: int. Num of samples of the GP values. 
        mean: array or array like of dimension of the batch in input. If None then 0's.  
        '''  
        if mean is None:
            mean = np.zeros(len(input))
        cov = self.kernel_fn(input, mode = mode)        
        cov = (cov + cov.T) /2
        return np.random.multivariate_normal(mean, cov, num_of_samples).T


    def generate_RKHS_samples(self, num_of_samples, num_of_fixed_points, dimension, mode = 'gradient_mse_ensemble', reg = 0):
        '''
        Generates function from RKHS corresponding to kernel_fn in ntk mode using the following mechanism:
        Sample num of fixed points on sphere. Sample GP values distributed according to the ntk at the given fixed points.
        Use the above samples as prior to a GP process distributed according to ntk kernel, and return the posterior mean
        function of this process and its covariance function. The mean is in the RKHS.  
        
        Input:
        num_of_samples: int.
        num_of_fixed_points: the number of points used for the prior. 
        dimension: the dimension of the sphere. 
        '''
        sample_functions = []
        train_data = []
        for i in range(num_of_samples):
            train_xs = gen_rand_data_on_sphere(num_of_fixed_points, dimension) 
            train_ys = self.generate_gp_samples(train_xs)
            train_data.append((train_xs, train_ys))
            sample_functions.append(self.train_with_kernel(train_xs, train_ys, mode, diag_reg = reg))
        return train_data, sample_functions


    def train_with_kernel(self, train_xs, train_ys, method = 'gradient_mse_ensemble', diag_reg = 0):
        ''' 
        To contain all methods to generate a function based on neural tangents API that calculates 
        GP posteriors after training on priors train_xs, train_ys. 
        TODO: add support for GO posterioir method in addition to gradient descent and compare the two. 
        Input:
        train_xs, train_ys: training data. 
        method: string (for now supports gradient_mse_ensemble).
        diag_reg: non-negative float to regularize the diagonal of the kernel matrix. 
        ''' 
        if method == 'gradient_mse_ensemble':
            return nt.predict.gradient_descent_mse_ensemble(kernel_fn = self._kernel_fn, x_train = train_xs, 
                                                      y_train = train_ys, diag_reg = diag_reg)
        
        elif method == 'gp_inference':
            def predict_fn(x_test, get, compute_cov = False):
                if get != 'ntk':
                    raise NotImplementedError
                else: 
                    k_test_train = self.kernel_fn(x_test, predict_fn.train_x, mode = None)
                    k_test_test = self.kernel_fn(x_test, x_test, mode = None)
                    return predict_fn.predict_fn(get = 'ntk', k_test_train = k_test_train, k_test_test = k_test_test)
            
            k_train_train = self.kernel_fn(train_xs, train_xs, mode = None)
            predict_fn.train_x = train_xs.copy()
            predict_fn.predict_fn = nt.predict.gp_inference(k_train_train, train_ys, diag_reg)
            return predict_fn


    def sample_from_predict_fn(self, predict_fn, dimension, noise_var = 0, xs = None, num_of_points = None):
        '''
        Method that gets a ground truth function predict_fn - according to neural tangents api 
        and samples values from this function.
        Input:
        num_of_points: positive integer. The number of points to sample from predict_fn. 
        predict_fn: function that receives x_test, get, compute_cov variables in signature.
        dimension: input dimension (must correspond to the one expected by predict_fn) 
        noise_var: non-ngative float to determine the noise added to the samples.    
        '''

        assert(xs is not None or num_of_points is not None)
        if xs is None:
            xs = gen_rand_data_on_sphere(num_of_points, dimension)
        ys, cov = predict_fn(x_test = xs, get = 'ntk', compute_cov = True)
        assert(not np.isnan(ys).any())
        if noise_var > 0:
            ys = add_Gaussian_noise(ys, np.sqrt(noise_var))
        return xs, ys, cov          
    

    def calculate_prediction_error(self, num_of_samples, dimension, predict_fn, true_fn):
        '''
        Gets predict function and ground truth function, generates random points in the domain of the functions
        and calculates the mse between the ground truth and prediction.
        Input:
        num_of_samples: positive integer - the number of samples to generate for mse calculation. 
        dimension: positive integer. dimension of the input - must correspond to what is expected by predict_fn and true_fn. 
        predict_fn, true_fn: pointers to the predict and true functions correspondingly. For signature see 
        'sample_from_predict_fn' documentation.
        '''
        xs, ys, _ = self.sample_from_predict_fn(true_fn, dimension, num_of_points = num_of_samples)
        predict_ys, _ = predict_fn(x_test = xs, get = 'ntk', compute_cov = True)
        return ys - predict_ys

    def conduct_experiment(self,num_of_experiments, num_of_training_points, num_of_fixed_points_for_RKHS, 
                                dimension, num_of_testing_samples, noise_var = 0, mode = 'gradient_mse_ensemble', 
                                reg_for_ground_truth =0, reg_for_training = 0, normalize_target_fn = False):
        '''
        This function generates ground truth functions and trains ntk based on labels from these ground truth,
        selecting the labelled data points randomly.
        params:
        num_of_experiments: num of experiments to do (will regulate the number of ground truth functions generated).
        num_of_fixed_points_for_RKHS: num of conditioning points for ground truth.
        num_of_testing_samples: num of points to calculate errors.
        noise_var: noise variance for the labels.
        reg_for_ground_truth: kernel regularization when generating ground truth,
        reg_for_training: kernel regularization for training. 
        '''
        train_data, sample_functions = self.generate_RKHS_samples(num_of_samples = num_of_experiments, 
                                num_of_fixed_points = num_of_fixed_points_for_RKHS, dimension = dimension, 
                                mode = mode, reg = reg_for_ground_truth)
        if normalize_target_fn:
            for i, funct in enumerate(sample_functions):
                norm = self.norm_fn(x = train_data[i][0], y = train_data[i][1], 
                                    reg = reg_for_ground_truth)
                def target_fn_norm(x_test, get = 'ntk', compute_cov = True):
                    mean, cov = funct(x_test = x_test, get = get, compute_cov = True) 
                    return mean/norm, cov / (norm **2)
                sample_functions[i] = target_fn_norm
        errors = []
        
        for func in sample_functions:
            train_x, train_y, _ = self.sample_from_predict_fn(num_of_points = num_of_training_points, 
                                        predict_fn = func, dimension = dimension, noise_var = noise_var)
            predict_fn = self.train_with_kernel(train_x, train_y, method = mode, diag_reg = reg_for_training)

            errors.append(self.calculate_prediction_error(num_of_testing_samples, dimension, 
                                                        predict_fn = predict_fn, true_fn = func))
        return errors

    def get_range_from_predict_fn(self, predict_fn, dimension, num_of_samples = 10000):
        xs, ys, _ = self.sample_from_predict_fn(predict_fn, dimension, num_of_points = num_of_samples)
        ys = np.expand_dims(np.squeeze(ys), axis = 0)
        sup = get_sup((ys + np.abs(ys))/2)
        inf = -get_sup((np.abs(ys) - ys)/2)
        return sup[0] - inf[0]

    def get_highest_variance_from_predict_fn(self, xs, predict_fn = None, mode = 'ntk'):
        '''
        Params: 
        xs: array of points. 
        predict_fn: prediction function of neural tangents predict_fn signature (see the call).
        mode: neural tangents modes. Use 'ntk' (as default).
        
        Calculates the posterir according to predict_fn on the set xs (mean and cov) and returns 
        the point in xs with the highest posterior variance. 
        '''
        if predict_fn is not None:
            predict_ys, cov = predict_fn(x_test = xs, get = 'ntk', compute_cov = True)
        else:
            cov = self.kernel_fn(xs, mode = mode)
        assert(not np.isnan(cov.any()))
        variance = np.diag(cov)
        arg_max = np.argmax(variance)
        return arg_max, xs[arg_max], variance[arg_max]

    def norm_fn(self, x, y, reg):
        kernel = self.kernel_fn(x)
        ker_reg = kernel + reg * np.identity(x.shape[0]) 

        inv_ker = np.linalg.inv(ker_reg)
        inv_ker_squared = np.matmul(inv_ker, inv_ker)
        return quadratic_form(inv_ker, y) - reg * quadratic_form(inv_ker_squared, y) 

class DataAcquisitionModule:
    '''
    Class responsinble for data acquisition. 
    params:
    num_of_fixed_points_for_RKHS: the number of conditioning points for ground truth function. 
    nn: neural network according to architecture of which we are training. 
    dimension: the dimension of the input space. 
    mode: use the default for ntk
    normalize_target_fn: boolean specifying whether to normalize the ground truth function or not. 
    regularization: this is used in constructor only when normalizing the ground truth.
    '''
    def __init__(self, nn: DenseNN, num_of_fixed_points_for_RKHS: int, dimension: int, mode = 'gradient_mse_ensemble', normalize_target_fn = False, reg_target_fn = 0,  norm_range = False):
        self._nn = nn
        self._target_fn_train_data, sample_functions = nn.generate_RKHS_samples(num_of_samples = 1, 
                                num_of_fixed_points = num_of_fixed_points_for_RKHS, dimension = dimension, mode = mode, reg = reg_target_fn)
        
        self._dimension = dimension
        self._regularization = reg_target_fn
        
        
        norm = nn.norm_fn(x = self._target_fn_train_data[0][0], y = self._target_fn_train_data[0][1], reg = 0)
        self._norm = norm
        if norm_range:
            norm = nn.get_range_from_predict_fn(predict_fn = sample_functions[0], dimension = self._dimension) 


        if normalize_target_fn:
            def target_fn_norm(x_test, get = 'ntk', compute_cov = True):
                mean, cov = sample_functions[0](x_test = x_test, get = get, compute_cov = True) 
                return mean/norm, cov / (norm **2)
            self._target_fn = target_fn_norm
        else:
            self._target_fn = sample_functions[0]
        
        self._range = nn.get_range_from_predict_fn(predict_fn = self._target_fn, dimension = self._dimension) 
                   

    def run_with_bagging(self, no_batches, points_in_batch, data_acquisition_iterations, noise_level = 0, diag_reg_for_training = 0, 
                mode = 'gradient_mse_ensemble', no_points_for_error_calc = 10000, auto_regularize = False, auto_regularization_factor = 0.01):
        ''' 
        This is the procedure to be used when resampling the set of points from which the data is acquired
        at each step and having a cap on the number of points used to calculate the posterior (this is necessary
        if we want posterioirs on large number of points). 

        Params: 
        no_batches: The number of batches to evaluate posterioir variance upon. 
        points_in_batch: number of points in each batch (effectively this is the maximum dimension of empirical
                            Gram matrices / posteriors calculated).
        data_acquisition_iterations: Number of training iterations / times the data is acquired. 
        noise_level: variance of Gaussian noise added to the training labels. 
        diag_reg_for_training: regularization for kernel training. 
        mode: leave default for ntk (neural tangents modes).
        no_points_for_error_calc: the number of points sampled for which errors will be calculated - the errors
                                being the difference between ground truth values and trained values. 
        '''
        if auto_regularize:
            noise_level = auto_regularization_factor * self._range
            diag_reg_for_training = noise_level

        xs = np.empty(shape = (data_acquisition_iterations, self._dimension))
        ys = np.empty(shape = (data_acquisition_iterations,1))
        predict_fn = None
        errors = []
        no_iter_error_calc = math.floor(no_points_for_error_calc / points_in_batch)

        for i in range(data_acquisition_iterations):
            print('iteration', i)
            curr_var = 0
            for j in range(no_batches):
                train_x = gen_rand_data_on_sphere(points_in_batch, dimension = self._dimension)
                _,train_y,_ = self._nn.sample_from_predict_fn(self._target_fn, self._dimension, 
                                                   noise_var = noise_level, xs = train_x)
                assert(not np.isnan(train_y).any())
                arg_max,x,var = self._nn.get_highest_variance_from_predict_fn(train_x, predict_fn = predict_fn)
                
                if var >= curr_var:
                    xs[i] = train_x[arg_max]
                    ys[i] = train_y[arg_max]
            predict_fn = self._nn.train_with_kernel(xs[0:i+1,:], ys[0:i+1], 
                        method = mode, diag_reg = diag_reg_for_training)
            
            #error calculation
            error = []
            for k in range(no_iter_error_calc):
                
                error.append(self._nn.calculate_prediction_error(num_of_samples = points_in_batch, 
                                                    dimension = self._dimension, 
                                                    predict_fn = predict_fn, true_fn = self._target_fn))
            error = np.array(error).flatten() 
            if no_iter_error_calc * points_in_batch < no_points_for_error_calc:
                residual = self._nn.calculate_prediction_error(num_of_samples = no_points_for_error_calc - 
                                                            (no_iter_error_calc * points_in_batch), 
                                                            dimension = self._dimension, 
                                                            predict_fn = predict_fn, true_fn = self._target_fn)
                error = np.concatenate((error, residual.squeeze()), axis = 0)

            errors.append(error)
        return errors, xs, ys



    def run(self, training_set, num_of_iterations, noise_level, diag_reg_for_training = 0, 
            mode = 'gradient_mse_ensemble', num_of_samples_for_error_calculation = 10000):
        '''
        Same as the run_with_bagging, except that a fixed data training set is given in training_set param. 
        '''
        train_x = training_set.copy()
        _,train_y,_ = self._nn.sample_from_predict_fn(self._target_fn, self._dimension, 
                noise_var = noise_level, xs = train_x)
        
        xs = np.empty(shape = (num_of_iterations, self._dimension))
        ys = np.empty(shape = (num_of_iterations,1))
        predict_fn = None
        errors = []

        for i in range(num_of_iterations):
            arg_max,x,var = self._nn.get_highest_variance_from_predict_fn(train_x, predict_fn = predict_fn)
            xs[i] = train_x[arg_max]
            ys[i] = train_y[arg_max]
            
            predict_fn = self._nn.train_with_kernel(xs[0:i+1,:], ys[0:i+1], 
                        method = mode, diag_reg = diag_reg_for_training)
             
            train_x = np.delete(train_x, arg_max, axis = 0)
            train_y = np.delete(train_y, arg_max, axis = 0)

            error = self._nn.calculate_prediction_error(num_of_samples = num_of_samples_for_error_calculation, 
                                                    dimension = self._dimension, 
                                                    predict_fn = predict_fn, true_fn = self._target_fn)
            errors.append(error)
        return errors, xs, ys

    
    def train_with_random_data(self, no_data_points_list: list, noise_level = 0, diag_reg_for_training = 0, 
                                mode = 'gradient_mse_ensemble', num_of_samples_for_error_calculation = 10000, 
                                auto_regularize = False, auto_regularization_factor = 0.01):
        errors = []
        if auto_regularize:
            noise_level = (auto_regularization_factor * self._range)**2
            diag_reg_for_training = noise_level

        for no_data_points in no_data_points_list:
            train_x = gen_rand_data_on_sphere(no_data_points, dimension = self._dimension)
            _,train_y,_ = self._nn.sample_from_predict_fn(self._target_fn, self._dimension, 
                            noise_var = noise_level, xs = train_x)
            assert(not np.isnan(train_y).any())
            predict_fn = self._nn.train_with_kernel(train_x, train_y, 
                        method = mode, diag_reg = diag_reg_for_training)
            error = self._nn.calculate_prediction_error(num_of_samples = num_of_samples_for_error_calculation, 
                                                    dimension = self._dimension, 
                                                    predict_fn = predict_fn, true_fn = self._target_fn)
            errors.append(error)
        return np.array(errors)

def relu_pow_s(s, cov12, x1, x2):
    prod = jnp.sqrt(jnp.where(x1*x2 >=0, x1*x2, 0))
    u = cov12/jnp.maximum(prod, 1e-12)

    u_tmp = jnp.minimum(u,1)
    u_tmp = jnp.maximum(u_tmp, -1)
    theta = jnp.arccos(u_tmp)
    
    if s==0:
        res = jnp.pi - theta
    if s==1:
        res = (jnp.sin(theta)+(jnp.pi - theta)*u)*prod

    if s==2:
        res = (3*jnp.sin(theta)*jnp.cos(theta) + (jnp.pi - theta)*(1+2*jnp.power(jnp.cos(theta),2)))*(prod**s)
    elif s==3:
        res = (15*jnp.sin(theta) - 11*jnp.power(jnp.sin(theta), 3) + (jnp.pi - theta)*(9*jnp.cos(theta)+ 6*jnp.power(jnp.cos(theta),3)))*(prod**(s))
    elif s==4:
        res = (9*jnp.sin(2*theta)-2.25*jnp.sin(4*theta)
                -44*jnp.power(jnp.sin(theta),3)*u + 
                24*(u**3)*jnp.sin(theta)+(jnp.pi-theta)*(36*jnp.cos(2*theta)+24*(u**4)+45))*(prod**s)
    return res/(2*jnp.pi) 



     
   