# CRESt
Copilot for Real-world Experimental Scientist (CRESt), an automated robotic platform for conducting electrochemical reactions at MIT

The functionality of each file is listed as below:

1. We leverage CallingGPT to control all functions for the automated electrochemical platform. These files could be located in CallingGPT, and interface.
2. The active learning algorithm files could be located in active_learning.
3. Automated scanning electron miscroscopy control and image analysis could be located in crest.
4. Database management leveraging the SQL database could be located in db_control.
5. The Opentrons liquid handling robot control files could be located in liquid_handling_robot.
6. The Xarm for automated electrochemical testing files could be located in robotic_testing.
7. The automation of controlling the BioLogic software for electrochemical testing with various testing protocols and analysis methods could be located in software_control.

For running the active learning algorithms, the users could follow the instructions in active_learning/main/README to run three different active learning algorithms. For controlling the hardware, the users need to have these hardware in the lab or install virtual machines to run the codes. 
