# CRESt
Copilot for Real-world Experimental Scientist (CRESt), an automated robotic platform for conducting electrochemical reactions at MIT

The functionality of each file is listed as below:

We leverage CallingGPT to control all functions for the automated electrochemical platform. These files could be located in CallingGPT, and interface.

The active learning algorithm files could be located in active_learning.

Automated SEM control and image analysis could be located in crest.

Database control leveraging the SQL database could be located in db_control.

The hardware control includes the control of the Opentrons liquid handling robot in liquid_handling_robot, Xarm for automated electrochemical testing in robotic_testing, , and database recording in db_control. For controlling the hardware, the users need to have these hardware in the lab or install virtual machines to run the codes. 
