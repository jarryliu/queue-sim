# queue-sim
A simulation program to simulate customized queuing models. 

- sim directory contains the basic components for simulation, including, job generation, queue, token bucket, server and so on.
- app directory contains examples to connect the sim components for running simulation for different topologies. 
   - token_test.py is the major app used to run the simulation for token bucket system. 
   - servers_test.py is the major app used to run the simulation for token bucket- server tandem system. 
- run directory contains some automted scripts to run the simulation with different settings (parameters, topology), the scripts will save some log files in local directory.
- plot directory contains the scripts plotting the simulation/theory figures.
   - draw.py/draw_2.py is used to plot the theory result figures
   - file_process.py parses the log files from run directory and plots the simulation results 
