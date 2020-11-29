Mollier_hX Diagram Generator
============================

The Mollier diagram is a graphic representation of the relationship
between air temperature, moisture content, and enthalpy. The diagram is
plotted in a skewed coordinate system which is chosen in order to
increase the reading accuracy for the unsaturated region of humid air.
Consequently, the x-axis is rotated clockwise until the isotherm t = 0
°C in the unsaturated region of humid air is horizontal. The isenthalp
lines run from top left to bottom right. The lines of constant water
content X run vertically. This script creates Mollier diagrams for a
variable pressure as well as with variable boundaries for both, water
load and temperature. Here, general usage and handling are introduced.

.. image:: mollier_hx_start_files/mollier_hx_p1_X0.0-0.08_T-15.0-59.9.png
   :width: 750


Getting started
---------------

The generation of the diagram is a two-step process. Thereby, the first
step is to generate the necessary data which means to solve the
thermodynamic states for a certain water load and temperature. In the
second step, the diagram itself is created. This allows a later
modification of the diagram without the necessity to recreate the data.
For both steps certain parameters have to be fined:

+-------------+-------+------------------------------------------------+
| key         | dtype | description                                    |
+=============+=======+================================================+
| pressure:   | float | pressure of air, vapor, and water mixture      |
|             |       | [bar] (default: 1.0)                           |
+-------------+-------+------------------------------------------------+
| X_start:    | float | initial waterload in [g/kg] (default: 0.0)     |
+-------------+-------+------------------------------------------------+
| X_end:      | float | final waterlaod in [g/kg] (default: 80.0)      |
+-------------+-------+------------------------------------------------+
| T_start:    | float | initial temperature in Grad Celcius (default:  |
|             |       | 0.0)                                           |
+-------------+-------+------------------------------------------------+
| T_end:      | float | final temperature in Grad Celcius (default:    |
|             |       | 60.0)                                          |
+-------------+-------+------------------------------------------------+
| temp:       | list  | list of float temperature values, their lines  |
|             |       | will be shown in the diagram (default: None)   |
+-------------+-------+------------------------------------------------+
| phi:        | list  | list of float phi values, their lines will be  |
|             |       | shown in the diagram (default: None)           |
+-------------+-------+------------------------------------------------+
| enthalpy:   | list  | list of float enthalpy values, their lines     |
|             |       | will be shown in the diagram (default: None)   |
+-------------+-------+------------------------------------------------+
| save_data:  | bool/ | if bool data is saved in current working dir,  |
|             | str   | if str(=path) data is saved at this location   |
|             |       | (default: None)                                |
+-------------+-------+------------------------------------------------+
| load_data:  | str   | the path where to find data (default: None)    |
+-------------+-------+------------------------------------------------+
| line_angle: | int   | correction angle to influence terminate line   |
|             |       | (if automatic correction fails) (default: 0)   |
+-------------+-------+------------------------------------------------+

All variables have a defined default so that the “standard” Mollier
diagram is created (pressure = 1bar, temperature = (0, 60), water load =
(0, 80)). In the case of the displayed lines with constant temperature,
phi, and enthalpy the following default behavior is defined:

-  temperature: lines of constant temperature will be displayed starting
   with T_start (round to 10¹) and continued with a step size of 10
   degrees. At 0 degrees, the lines for ice as well as liquid are
   included.
-  phi: lines of constant phi are displayed starting with 0.0 until 1.0
   with a step size of 0.1
-  enthalpy: Levels of constant enthalpy are displayed with a step size
   of 5 over the entire visible space. However, labels are only given
   for every fourth line.

Every diagram is saved automatically in the current working directory.
As a result, the diagram can be generated as follows:

.. code:: python

    %matplotlib inline
    from stfs_pytoolbox.mollier_hx import Mollier
    
    diagram = Mollier(pressure=10, T_end=120, X_end=30)


.. parsed-literal::

    Data Generation: 100%|██████████| 1151/1151 [05:19<00:00,  3.60it/s]



.. image:: mollier_hx_start_files/mollier_hx_start_1_1.png
   :width: 650


Plot modification
-----------------

It is possible that due to higher pressures or a variation of other
parameters, the diagram does not look as expected. In order to prevent
that the whole data space has to be recreated, the script allows an easy
modification of the parameters presented in the previous section. As an
example, we want to consider our created diagram since the “terminate
line” where normally temperature, as well as enthalpy lines, end, it not
visible. The script includes a function that controls the angle of the
terminate line, however, such errors can occur. In order to solve this
issue, the special flag “line_angle” can be used. Furthermore, in this
example, the phi lines should be restricted to certain values. With the
aim to recreate just the plot with the mentioned differences, the
function “create_plot” is employed as follows:

.. code:: python

    diagram.create_plot(phi=[0.1, 0.3, 0.7, 1.0], line_angle=12)



.. image:: mollier_hx_start_files/mollier_hx_start_3_0.png
   :width: 650


Nevertheless, the boundaries of the diagram cannot be changed with a
flag in the create_plot function. Instead, the parameters have to be
changed directly, followed by an execution of the create_plot function.
Thereby, the following rules have to be fulfilled:

-  While temperatures are saved in Celcius, the water load is
   transformed in [kg/kg], which means that there is a difference
   between the input in the initialization of the class and the later
   change of the parameter
-  Flag of the final temperature in the diagram is “T_max” not “T_end”
   since “T_end” can extent the chemical calculatable maximum
   temperature
-  When the boundaries are changed to values outside the calculated
   thermochemical states, the function “generate_data” has to be
   executed otherwise the process will fail!

In the following, an example redefining the temperature and water load
boundaries is shown. However, no additional data has to be created.

.. code:: python

    diagram.T_max = 100
    diagram.X_end = 25 * 1e-3
    
    diagram.create_plot()



.. image:: mollier_hx_start_files/mollier_hx_start_5_0.png
   :width: 650


--------------

Author: Pascal Roth

E-Mail: roth.pascal@outlook.de
