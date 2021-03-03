# ICM-DQN
An implementation of a modified DQN using an ICM playing SC2 using PYSC2

Requiremnt: 
	Python 3 (tested with 3.7)
	pysc2 (tested with 3.0.0)
	tensorflow (tested with 2.1.0)
	StarCraft II + Maps


Running the agent
1. change director to ../ICM+DQN_v2.08
2. python -m run --map DefeatRoaches --max_episode 10000 --agent agents.deepq.DQNAgent --training True
	a. --map DefeatRoaches, changes the the map
	b. --max_episode 10000, determines how many episodes should be run
	c. --agent agents.deepq.DQNAgent, determines which agent should be use
	d. --training True, determines if the agent will train or validate
	e. --save_every_nth_episode 1, determines how often the agent will save its progress


Running an already trained agent:
1. For runnig the agent, it needs 3 folders with 4 files in each. (If your running the agent, then the agent will save its progress after each episode, in the checkpoints-folder)
2. Paste these 3 folders, containing the files in the "checkpoints" folder
3. Run the agent like usual. 


To Tensorboard Graphs:
	1. Open Anaconda Prompt
	2. tensorboard --logdir tensorboard/deepq_statistics
	3. Open up webbrowser
	4. Enter "http://localhost:6006/" in url


