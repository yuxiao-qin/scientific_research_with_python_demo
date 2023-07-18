# scientific_research_with_python_demo
This is a demo of how to do simulation in scientific research with python. 
The latest function of perriodogram aims at singal arc estimation.

## Run a demo of periodogram
* To run a demo of periodogram estimation , you need read and run [periodogram_demo.py](./template/periodogram_demo.py)
* This can caculate paramsmeters per arc once you have  provided the initial parameters such as Nifg , noise_level( SNR ) , v_orig , h_orig,Num_search , search_step.

## Notice
* The timebaseline and normal baseline have already been set randomly. Instead of entering a parameter, you should modify them directly.
* The threshold for determining whether the parameter estimates are correct or not has also been set.But you also can change them.
* For periodogram parameter estimation, it is necessary to solve the problem of evaluating the quality of parameter estimation. 
    At present, for some experiments, some random parameters (normal baseline guass_nosie, etc.) are set, 
    and a rough success rate of parameter estimation can be obtained by repeating parameter estimation 100 times. 
    But this can cause the code to take too long to run.
- ### Example inputs
- #### initial parameters
{
WAVELENGTH = 0.056  # [unit:m]
Nifg = 10
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 50
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
Num_search1_max = 200
Num_search1_min = 80
Num_search2_max = 1300
Num_search2_min = 300
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
}
- ### Example output
---------------------------------------------------------------------
{'height': 30.007500000000004, 'velocity': 0.049868899999999994}
{'height': 30.0015, 'velocity': 0.05006090000000001}
{'height': 30.005300000000002, 'velocity': 0.0500889}
{'height': 30.0089, 'velocity': 0.04988889999999999}
{'height': 29.9984, 'velocity': 0.05004310000000001}
{'height': 30.009600000000002, 'velocity': 0.05004290000000001}
{'height': 29.9983, 'velocity': 0.0499639}
{'height': 29.9955, 'velocity': 0.04991110000000001}
{'height': 30.0034, 'velocity': 0.050002899999999996}
{'height': 29.992199999999997, 'velocity': 0.04987110000000001}
----------------------------------------------------------------------

## Experiments of deformation rate v
* To run a demo of periodogram estimation , you need read and run [lab_v.py](./template/lab_v.py)
* This can caculate paramsmeters per arc once you have  provided the initial parameters such as Nifg , SNR , v_orig , h_orig,Num_search , search_step.

- ### Notice
* You need to use the 'np.linspace' to get an array of deformation rate 'v_orig' based on fixed step.
* When using funtions of data-plot , rememmer that you need make sure your file path and filename in order to avoid losting the pictures you draw .
* The folder [plot](./scientific_research_with_python_demo/plot/) is used to store the resulting images
* The folder [data_save](./scientific_research_with_python_demo/data_save/) is used to store the related data (etimated parameters and success rate)
* You can change the [function](./scientific_research_with_python_demo/data_plot.py) to get the data graph you want

- ### Example inputs
    {
    WAVELENGTH = 0.056  # [unit:m]
    Nifg = 10
    h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
    noise_level = 80
    step_orig = np.array([1.0, 0.0001])
    param_orig = np.array([0, 0])
    param_name = ["height", "velocity"]
    v_orig = np.linspace(0.001, 0.2, 200)
    h = h_orig
    std_param = np.array([40, 0.08])
    Num_search1_max = 80
    Num_search1_min = 80
    Num_search2_max = 1600
    Num_search2_min = 300
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    success_rate = np.zeros(len(v_orig))
    }

- ### Example output
    [Bar plot : changing defomation rate](./scientific_research_with_python_demo/plot/snr_v_test5.png)
    [Line plot:changing defomation rate](./scientific_research_with_python_demo/plot/snr_v_test6.png)

## Experiments of resudual height h
This is similar to the experiments of deformation rate v . 
You need to use the 'np.linspace' to get an array of resudual height 'h_orig' based on fixed step.
- ### Example inputs
    {
    WAVELENGTH = 0.056  # [unit:m]
    Nifg_orig = np.linspace(10, 50, 41, dtype=int)
    v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
    h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
    noise_level = 70
    step_orig = np.array([1.0, 0.0001])
    std_param = np.array([40, 0.01])
    param_orig = np.array([0, 0])
    param_name = ["height", "velocity"]
    v = v_orig
    h = h_orig
    Num_search1_max = 80
    Num_search1_min = 80
    Num_search2_max = 1600
    Num_search2_min = 300
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    success_rate = np.zeros(len(Nifg_orig))
    est = np.zeros((81, 100))
    }

- ### Example output
[Bar plot](./scientific_research_with_python_demo/plot/Nifg_70.png)

-------------------------------------------------------------
## Experiments of the number of interference images
This is similar to the experiments of deformation rate v . You need to use the 'np.linspace' to get an array of 'Nifg' based on fixed step.
## Experiments of Revisit cycle dt
* To run a demo of periodogram estimation , you need read and run [lab_dt.py](./template/lab_dt.py)
* This can caculate paramsmeters per arc once you have  provided the initial parameters such as Nifg , SNR , v_orig , h_orig,Num_search , search_step, dt.

### Notice
* The timebaseline have been set already , you need multiply this argument by dt which is a factor to change revisit cycle .
* You need to use the 'np.linspace' to get an array of dt .
    example:
    ' time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg) * dt '

- ### Example inputs
    {
    WAVELENGTH = 0.0056  # [unit:m]
    Nifg = 10
    v_orig = 0.05  # [m]
    h_orig = 30  # [m]
    noise_level = 70
    dt_orig = np.linspace(0.1, 1.0, 10)
    step_orig = np.array([1.0, 0.0001])
    param_orig = np.array([0, 0])
    param_name = ["height", "velocity"]
    v = v_orig
    h = h_orig
    std_param = np.array([40, 0.08])
    Num_search1_max = 80
    Num_search1_min = 80
    Num_search2_max = 1600
    Num_search2_min = 300
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    success_rate = np.zeros(len(dt_orig))
    }

- ### Example output
[Bar plot](./scientific_research_with_python_demo/plot/test_dt.png)

------------------------------------------------------------


