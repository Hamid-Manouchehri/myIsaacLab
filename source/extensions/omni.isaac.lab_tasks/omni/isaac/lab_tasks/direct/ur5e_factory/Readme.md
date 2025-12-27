## Tips:
For training and playing the env with an RL workflow (e.g. rl_games) in `Direct` paradigm for any task (e.g. Isaac_Factory_PegInsert_Direct_v0), first change the directory to the IsaacLab dir, then:

Trainig (windows):
```
isaaclab.bat -p source\standalone\workflows\rl_games\train.py --task Isaac_Factory_PegInsert_Direct_v0 --headless --video
```

Playing (windows):
```
isaaclab.bat -p source\standalone\workflows\rl_games\play.py --task Isaac_Factory_PegInsert_Direct_v0 --num_envs 128 --use_last_checkpoint
```

Visualizing while training:
If you want to see the GUI while training the robot; how the policy and env behaves and whether you have correctly set up Isaac Lab or not:
```
isaaclab.bat -p source\standalone\workflows\rl_games\train.py --task Isaac_Factory_PegInsert_Direct_v0 --num_envs 20
```