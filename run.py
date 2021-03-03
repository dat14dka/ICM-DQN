#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.") #CHANGED from True to False
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 10000, "Total episodes.") #10000, 30 takes 20ish minutes
flags.DEFINE_integer("step_mul", 20 , "Game steps per agent step.") # CHANGED from 8

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_integer("save_every_nth_episode", 1,
                    "saves every Nth episode")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.") #True

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")

# agent specific hyperparameters and settings
flags.DEFINE_float("learning_rate",  1e-5, "Learning rate.") 
flags.DEFINE_float("discount_factor", 0.95, "Future reward discount factor.") 
flags.DEFINE_bool("training", True, "Train the model during the run.")
flags.DEFINE_string("save_dir", None, "Where to save tensorflow ckpts.")
flags.DEFINE_string("ckpt_name", None, "Name for ckpt files.")
flags.DEFINE_string("summary_path", None, "Where to write tensorboard summaries.")

# DQNMoveOnly
flags.DEFINE_float("epsilon_max", 1.0, "Maximum exploration probability.")
flags.DEFINE_float("epsilon_min",  0.0, "Minimum exploration probability.")
flags.DEFINE_integer("epsilon_decay_steps", 15000 , "Linear epsilon decay steps.")
flags.DEFINE_integer("train_frequency", 1, "How often to train network.")
flags.DEFINE_integer("target_update_frequency", 500, "How often to update target network.") # was 500
flags.DEFINE_integer("max_memory", 1000, "Experience replay buffer capacity.")
flags.DEFINE_integer("batch_size", 8, "Training batch size.") # changed from 16
flags.DEFINE_bool("indicate_nonrandom_action", False, "Show nonrandom actions.")

# A2CAtari
flags.DEFINE_integer("trajectory_training_steps", 40, "When to cut trajectory and train network.")
flags.DEFINE_float("value_gradient_strength", 0.5, "Scaling parameter for value estimation gradient.")
flags.DEFINE_float("regularization_strength", 0.01, "Scaling parameter for entropy regularization.")


def run_thread(agent_classes, players, map_name, visualize):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(map_name=map_name,
                        players=players,
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=FLAGS.feature_screen_size,
                            feature_minimap=FLAGS.feature_minimap_size,
                            rgb_screen=FLAGS.rgb_screen_size,
                            rgb_minimap=FLAGS.rgb_minimap_size,
                            action_space=FLAGS.action_space,
                            use_feature_units=FLAGS.use_feature_units),
                        step_mul=FLAGS.step_mul,
                        game_steps_per_episode=FLAGS.game_steps_per_episode,
                        disable_fog=FLAGS.disable_fog,
                        visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agents = [agent_cls() for agent_cls in agent_classes]
        run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
        if FLAGS.save_replay:
            env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
    """Run an agent."""
    if FLAGS.trace:
        stopwatch.sw.enable()
        stopwatch.sw.trace()
    if FLAGS.profile:
        stopwatch.sw.enable()
    
    

    map_inst = maps.get(FLAGS.map)

    agent_classes = []
    players = []

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race]))

    if map_inst.players >= 2:
        if FLAGS.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                       sc2_env.Difficulty[FLAGS.difficulty]))
        else:
            agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
            agent_cls = getattr(importlib.import_module(agent_module), agent_name)
            agent_classes.append(agent_cls)
            players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race]))

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread,
                             args=(agent_classes, players, FLAGS.map, False))
        threads.append(t)
        t.start()

    run_thread(agent_classes, players, FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
