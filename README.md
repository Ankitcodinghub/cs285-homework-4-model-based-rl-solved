# cs285-homework-4-model-based-rl-solved
**TO GET THIS SOLUTION VISIT:** [CS285 Homework 4-Model-Based RL Solved](https://www.ankitcodinghub.com/product/cs285-deep-reinforcement-learning-hw4-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113013&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS285 Homework 4-Model-Based RL Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
1 Introduction

The goal of this assignment is to get experience with model-based reinforcement learning. In general, model-based reinforcement learning consists of two main parts: learning a dynamics function to model observed state transitions, and then using predictions from that model in some way to decide what to do (e.g., use model predictions to learn a policy, or use model predictions directly in an optimization setup to maximize predicted rewards).

In this assignment, you will do the latter. You will implement both the process of learning a dynamics model, as well as the process of creating a controller to perform action selection through the use of these model predictions. For references to this type of approach, see this paper and this paper.

2 Model-Based Reinforcement Learning

MBRL consists primarily of two aspects: (1) learning a dynamics model and (2) using the learned dynamics models to plan and execute actions that minimize a cost function (or maximize a reward function).

2.1 Dynamics Model

In this assignment, you will learn a neural network dynamics model fθ of the form

∆ˆ t+1 = fθ(st,at) (1)

which predicts the change in state given the current state and action. So given the prediction ∆ˆ t+1, you can generate the next prediction with

ˆst+1 = st + ∆ˆ t+1. (2)

See the previously referenced paper for intuition on why we might want our network to predict state differences, instead of directly predicting next state.

You will train fθ in a standard supervised learning setup, by performing gradient descent on the following objective:

L(θ) = X k(st+1 −st) − fθ(st,at)k22

(st,at,st+1)∈D (3)

= X (4)

(st,at,st+1)∈D

In practice, it’s helpful to normalize the target of a neural network. So in the code, we’ll train the network to predict a normalized version of the change in state, as in

L(θ) = X kNormalize(s . (5)

(st,at,st+1)∈D

Since fθ is trained to predict the normalized state difference, you generate the next prediction with

ˆst+1 = st + Unnormalize(fθ(st,at)). (6)

2.2 Action Selection

Given the learned dynamics model, we now want to select and execute actions that minimize a known cost function (or maximize a known reward function). Ideally, you would calculate these actions by solving the following optimization:

∞

a∗t = argminXc(ˆst0,at0) where ˆst0+1 = ˆst0 + fθ(ˆst0,at0). (7)

at:∞ t0=t

However, solving Eqn. 7 is impractical for two reasons: (1) planning over an infinite sequence of actions is impossible and (2) the learned dynamics model is imperfect, so using it to plan in such an open-loop manner will lead to accumulating errors over time and planning far into the future will become very inaccurate.

Instead, we will solve the following gradient-free optimization problem:

t+H−1

A∗ = arg min X c(ˆst0,at0) s.t. ˆst0+1 = ˆst0 + fθ(ˆst0,at0), (8)

{A(0),…,A(K−1)} t0=t

in which A ) are each a random action sequence of length

Additionally, since our model is imperfect and things will never go perfectly according to plan, we adopt a model predictive control (MPC) approach, in which we solve Eqn. 8 at every time step to select the best H-step action sequence, but then we execute only the first action from that sequence before replanning again at the next time step using updated state information.

Finally, note that the random-shooting optimization approach mentioned above can be greatly improved (see this paper).

2.3 On-Policy Data Collection

2.4 Ensembles

A simple and effective way to improve predictions is to use an ensemble of models. The idea is simple: rather than training one network fθ to make predictions, we’ll train N independently initialized networks , and average their predictions to get your final predictions

. (9)

In this assignment, you’ll train an ensemble of networks and compare how different values of N effect the model’s performance.

3 Code

You will implement the MBRL algorithm described in the previous section.

3.1 Overview

Obtain the code from https://github.com/berkeleydeeprlcourse/ homework_fall2020/tree/master/hw4.

You will add code to the following three files: agents/mb_agent.py, models/ff_model.py, and policies/MPC_policy.py. You will also need to edit these files by copying code from past homeworks or Piazza: infrastructure/rl_trainer.py and infrastructure/utils.py.

Problem 1

What you will implement:

What code files to fill in:

1. cs285/agents/mb_agent.py

2. cs285/models/ff_model.py

3. cs285/infrastructure/utils.py

4. cs285/policies/MPC_policy.py (just one line labeled TODO(Q1) for now)

What commands to run:

python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n500_arch1x32

–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 500 -n_layers 1 –size 32 –scalar_log_freq -1 –video_log_freq -1

python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n5_arch2x250

–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 5 -n_layers 2 –size 250 –scalar_log_freq -1 –video_log_freq -1

python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n500_arch2x250

–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 500 -n_layers 2 –size 250 –scalar_log_freq -1 –video_log_freq -1

Your code will produce plots inside your logdir that illustrate your model prediction error (MPE). The code will also produce a plot of the losses over time. For the first command, the loss should go below 0.2 by the iteration 500. These plots illustrate, for a fixed action sequence, the difference between your model’s predictions (red) and the ground-truth states (green). Each plot corresponds to a different state element, and the title reports the mean mean-squared-error across all state elements. As illustrated in the commands above, try different neural network architectures as well different amounts of training. Compare the results by looking at the loss values (i.e., itr 0 losses.png), the qualitative model predictions (i.e., itr 0 predictions.png), as well as the quantitative MPE values (i.e., in the title of itr 0 predictions.png).

What to submit: For this question, submit the qualitative model predictions (itr 0 predictions.png) for each of the three runs above. Comment on which model performs the best and why you think this might be the case.

Note that for these qualitative model prediction plots, we intend for you to just copy the png images produced by the code.

Problem 2

What will you implement:

Action selection using your learned dynamics model and a given reward function.

What code files to fill in:

1. cs285/policies/MPC_policy.py What commands to run:

python cs285/scripts/run_hw4_mb.py –exp_name q2_obstacles_singleiteration –env_name obstacles-cs285-v0 -add_sl_noise –num_agent_train_steps_per_iter 20 –n_iter 1 -batch_size_initial 5000 –batch_size 1000 –mpc_horizon 10

Recall the overall flow of our rl trainer.py. We first collect data with our policy (which starts as random), we then train our model on that collected data, and we then evaluate the resulting MPC policy (which now uses the trained model). To verify that your MPC is indeed doing reasonable action selection, run the command above and compare Train AverageReturn (which was the execution of random actions) to Eval AverageReturn (which was the execution of MPC using a model that was trained on the randomly collected training data). You can expect Train AverageReturn to be around -160 and Eval AverageReturn to be around -70 to -50.

What to submit:

Submit this run as part of your run logs, and include a plot of Train AverageReturn and Eval AverageReturn in your pdf. Note that these will just be single dots on the plot, since we ran this for just 1 iteration.

Problem 3

What will you implement:

MBRL algorithm with on-policy data collection and iterative model training.

What code files to fill in:

None. You should already have done everything that you need, because rl trainer.py already aggregates your collected data into a replay buffer. Thus, iterative training means to just train on our growing replay buffer while collecting new data at each iteration using the most newly trained model.

What commands to run:

python cs285/scripts/run_hw4_mb.py –exp_name q3_obstacles –env_name obstacles-cs285-v0 –add_sl_noise –num_agent_train_steps_per_iter 20 –batch_size_initial 5000 –batch_size 1000 –mpc_horizon 10 -n_iter 12

python cs285/scripts/run_hw4_mb.py –exp_name q3_reacher –env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size_initial 5000 -batch_size 5000 –n_iter 15

python cs285/scripts/run_hw4_mb.py –exp_name q3_cheetah –env_name cheetah-cs285-v0 –mpc_horizon 15 –add_sl_noise -num_agent_train_steps_per_iter 1500 –batch_size_initial 5000 -batch_size 5000 –n_iter 20

You should expect rewards of around -25 to -20 for the obstacles env (takes 40 minutes), rewards of around -250 to -300 for the reacher env (takes 2-3 hours), and rewards of around 250-350 for the cheetah env takes 3-4 hours. All numbers assume no GPU.

What to submit:

Submit these runs as part of your run logs, and include the performance plots in your pdf.

Problem 4

What will you implement:

You will compare the performance of your MBRL algorithm as a function of three hyperparameters: the number of models in your ensemble, the number of random action sequences considered during each action selection, and the MPC planning horizon.

What code files to fill in: None.

What commands to run:

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon5 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 5 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon15 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 15 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon30 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 30 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_numseq100 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15 -mpc_num_action_sequences 100

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_numseq1000 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15 -mpc_num_action_sequences 1000

python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble1 -env_name reacher-cs285-v0 –ensemble_size 1 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size

800 –n_iter 15 python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble3 -env_name reacher-cs285-v0 –ensemble_size 3 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size

800 –n_iter 15 python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble5 -env_name reacher-cs285-v0 –ensemble_size 5 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size

800 –n_iter 15

What to submit:

1) Submit these runs as part of your run logs.

2) Include the following plots (as well as captions that describe your observed trends) of the following:

• effect of ensemble size

• effect of the number of candidate action sequences

• efffect of planning horizon

Be sure to include titles and legends on all of your plots, and be sure to generate your plots by extracting the corresponding performance numbers from your saved tensorboard eventfiles.

Submission

3.2 Submitting the PDF

Your report should be a PDF document containing the plots and responses indicated in the questions above.

3.3 Submitting the Code and Logs

In order to turn in your code and experiment logs, create a folder that contains the following:

• The cs285 folder with all the .py files, with the same names and directory structure as the original homework repository (not include the data/ folder). A plotting script should also be submitted, which should be a python script (or jupyter notebook) such that running it can generate all plots from your pdf. This plotting script should extract its values directly from the experiments in your runlogs and should not have hardcoded reward values.

As an example, the unzipped version of your submission should result in the following file structure. Make sure that the submit.zip file is below 15MB and that they include the prefix hw4mb .

submit.zip

Turn in your assignment on Gradescope. Upload the zip file with your code and log files to HW4 Code, and upload the PDF of your report to HW4.
