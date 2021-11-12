import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import configparser
import ast
import itertools


### Config files parsers (modified to accpet comments)

class FileConfigParser(configparser.RawConfigParser):
    def __init__(self):
        configparser.RawConfigParser.__init__(self, inline_comment_prefixes=('#',';'))


def list_to_int(lista):
    lista= ast.literal_eval(lista)
    lista=[int(el) for el in lista]
    return lista

def list_to_float(lista):
    lista= ast.literal_eval(lista)
    lista=[float(el) for el in lista]
    return lista

#coordinates creation
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensor =  np.arange(-1, dim)
    print(tensor, tensor)
    mgrid =np.stack( np.meshgrid(tensor, tensor), -1)
    #mgrid = np.reshape(-1, dim)
    return mgrid


'''
In three following functions it is assumed that the input is 2 dim array and the aggregation is along the
columns
'''

def get_mse(values):
  return np.power(values,2).mean(axis = -1)

def get_mae(values):
  return np.abs(values).mean(axis = -1)

def get_sup(values):
  return np.max(np.abs(values), axis = -1)

def create_polynomial(power, constant_mul):
  def func(xs):
    return constant_mul*(xs ** power)
  return func

def plot_error_rate_function(func, x, y, mesh = 100, plt_obj = None, color = 'red'):
  min_x = np.min(x)
  max_x = np.max(x)

  x_points = np.linspace(min_x, max_x, mesh)
  y_points = func(x_points)
  
  if plt_obj is None:
    plt_obj = plt
  plt_obj.plot(x, y, 'ro', markersize=5, label='train', color = color)
  plt_obj.plot(x_points, y_points, color = 'purple', linestyle = 'dashed', label='polynomial fit')
  return plt_obj 
    
def fit_polynomial(x, y):
  x = np.log(x)
  y = np.log(y)

  res = stats.linregress(x,y)
  return x, y, res.slope, res.intercept, res.stderr
  
def generate_uniform_set_on_sphere(partition_mesh, dimension):
  stride = np.pi / partition_mesh

  partition_pi = np.array([i*stride for i in range(partition_mesh)] + [np.pi])
  partition_2pi = np.array([i*stride for i in range(partition_mesh * 2)])
  
  grid_count_pi = len(partition_pi) 
  grid_count_2pi = len(partition_2pi)
  cos_values = np.cos(partition_pi)
  sin_values = np.sin(partition_pi)


  cos_values_last_coord = np.cos(partition_2pi)
  sin_values_last_coord = np.sin(partition_2pi)

  assert(partition_mesh == len(partition_pi)-1)
  choice_set = np.arange(grid_count_pi)

  point_set = np.empty(shape = ((grid_count_pi ** (dimension - 2)) * grid_count_2pi ,dimension))

  count = 0
  combination_count = 0
  for choice in itertools.product(choice_set, repeat = dimension - 2):
    combination_count += 1
    partial_coordinate = []
    curr_point = 1
    for i in range(dimension - 2):
      partial_coordinate.append(curr_point * cos_values[choice[i]])
      curr_point *= sin_values[choice[i]]
    
    for i in range(grid_count_2pi):
      full_coordinate = partial_coordinate + [curr_point * cos_values_last_coord[i], curr_point * sin_values_last_coord[i]]
      arr = np.array(full_coordinate)
      assert(not np.any(np.isnan(arr)))
      point_set[count,:] = np.array(full_coordinate)
      count +=1 
  return point_set


def add_Gaussian_noise(mean, scale = 1):
  return np.random.normal(loc = mean, scale = scale)
  
def gen_rand_data_on_sphere(number_of_points, dimension):
    x = np.random.randn(number_of_points, dimension)
    for i, vec in enumerate(x):
        x[i] = vec/np.linalg.norm(vec)
    return x

def format_plot(x=None, y=None): 
  # plt.grid(False)
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

def plot_fn(train, test, *fs):
  train_xs, train_ys = train
  

  plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')
  
  if test != None:
    test_xs, test_ys = test
    plt.plot(test_xs, test_ys, 'ro', markersize=3, label='test')


    for f in fs:
      plt.plot(test_xs, f(test_xs), '-', linewidth=3)

  plt.xlim([-np.pi, np.pi])
  format_plot('$x$', '$f$')

  def plot_or_save_results(res, file_name, type_of_aggregation = 'mse', show = False):
    NUM_OF_FIXED_POINTS_FOR_RKHS = res.shape[0]
    DIMENSION = res.shape[1]
    NUM_OF_EXPERIMENTS = res.shape[2]
    pp = PdfPages('file_name' + '.pdf')

    if type_of_aggregation == 'mse':
      res = get_mse(res)
    elif type_of_aggregation == 'mae':
      res = get_mae(res)
    elif type_of_aggregation == 'sup':
      res = get_sup(res)

    for index in range(NUM_OF_FIXED_POINTS_FOR_RKHS):
        for i in range(DIMENSION):
          for experiment in range(NUM_OF_EXPERIMENTS):
              errors = res[index, i,experiment,:]
              xs = np.arange(start = start_error + 1, stop = len(errors)+1 + start_error)
              x,y,slope,intercept =  fit_polynomial(xs, errors)
                
              func = lambda x : slope * x + intercept
              pl = plot_error_rate_function(func, x, y)
              pl.title('Conditioning points:{}, Dimension:{}, Experiment: {}'.format(10*(index+1), (i+1)*2, experiment))
                #pl.xscale('log')
                #pl.yscale('log')
              ax = pl.gca()
              fig = pl.gcf()
              pl.text(0.5, 0.9, 'Slope: {} \n Intercept: {}'.format(slope,intercept), horizontalalignment = 'left', 
                    verticalalignment='center', transform=ax.transAxes)
              pp.savefig(fig)
              if show:
                pl.show()
    pp.close()

def quadratic_form(mat, vec):
  return np.matmul(vec.T, np.matmul(mat, vec))

###Loger

def logs(output_path, output_name, data_partition_mesh, dataset, data_dimensions, nn_output_dim, nn_hidden,num_of_fixed_points_for_RKHS, num_of_iterations, noise_level, diag_reg_for_training, xs, errors, function):
    
    log_dictionary={
               'dataset': dataset,
               'data_partition_mesh': data_partition_mesh,
               'data_dimensions': data_dimensions,
               'nn_hidden': nn_hidden,
               'nn_output_dim': nn_output_dim,
               'num_of_fixed_points_for_RKHS': num_of_fixed_points_for_RKHS,
               'num_of_iterations': num_of_iterations,
               'noise_level': noise_level,
               'diag_reg_for_training': diag_reg_for_training,
               'errors': errors,
               'xs': xs,
               'ys':ys,
               'function': function
               }

    output_file = output_path + output_name + '.pkl'
    with open(output_file, 'wb') as handle:
        pickle.dump(log_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)        
            
            

           

