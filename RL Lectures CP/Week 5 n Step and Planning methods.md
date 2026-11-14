
In this lesson, we are going to discuss a limitation of the one-step temporal-difference learning methods we have looked at so far, and introduce planning methods as a potential solution. The term _'planning'_ is used in many areas of AI to refer to different things, but, as we shall discuss, it has a very specific meaning in reinforcement learning.

The planning methods that we will introduce in this lesson allow an agent to learn a model environment based on its experience. This model may be used to generate further simulated experience, which can be used alongside our agent's real experience to perform learning.

# Limitations of One-Step Learning

Completion requirements

Consider the simple 5×5 gridworld illustrated below. Our agent starts at the initial state marked 'S' and must navigate to the goal state marked 'G'. The agent can move deterministically 'North', 'South', 'East' or 'West' one grid square each time step, and moving into a wall causes the agent to remain in its current grid square. Rewards are 0 for all transitions, except a reward of +10 for transitioning to the goal state (you may assume a discount factor 𝛾<1).

**A simple 5×5 gridworld.**

Let us train a Sarsa agent to reach the goal state in this simple gridworld. Assume that our agent is using an 𝜖-greedy policy, and its initial Q-value estimates are set to 0 for all state-action pairs. During its first episode of training, our agent takes the trajectory through the gridworld illustrated below.

**The path taken by our agent during its first episode of training in the simple 5×5 gridworld.**

---

### Stop and Think

##### Take a moment to think about the updates our agent will perform as it moves along this trajectory. What update does Sarsa perform after the first time-step (𝑡=0)? What update does Sarsa perform after the final time-step (𝑡=7)? Assume a learning-rate 𝛼=0.2 and a discount factor 𝛾=0.9.

In this first episode, our agent learned only about the action it performed immediately prior to reaching the goal state. The reward received at the end of the episode did not teach our agent anything about the actions it had taken earlier. The action values updated in each state are illustrated below.

**The action values updated using Sarsa in our 5x5 gridworld after a single episode of training.** 

Notice that only the value of the action taken directly before the end of the episode has been changed.

If our agent were to follow the same path through the environment again during a second episode of training, the learned values would propagate one time-step further back, as shown below. It is easy to imagine that it would take considerable additional training for these value changes to propagate all the way back to the starting state.

**The Q-values updated using Sarsa in our 5x5 gridworld after a second episode of training.**

Notice that information about the reward received when reaching the goal has now propagated back two steps away from the goal state.

There are many problems like this gridworld, where our agent only receives rewards at the end of an episode. For example, in two-player games like tic-tac-toe, we only reward or penalise our agent at the end of each game, depending on whether it has won, lost, or drawn. As was the case in our simple gridworld example, a one-step temporal-difference learning method like Sarsa will require many episodes of experience to help value changes near the terminal states eventually propagate backwards towards the initial states.

The methods that we will be looking at this week help these value changes to propagate through the state-space more quickly, and allow our agent to learn more efficiently from a given set of experience.

This week, we will be talking a lot about value changes propagating backwards from near-terminal states towards initial states. Please note that, despite the similar terminology being used, this has absolutely nothing to do with the backpropagation algorithm for training feedforward neural networks, which you may have encountered previously.

# Models in Reinforcement Learning

Completion requirements

The temporal-difference learning methods we have looked at so far in this unit have been **model-free**methods, which means they do not make use of a model of our agent's environment. We will now look at some simple **model-based** methods, which learn and use models of our agent's environment in order to improve learning efficiency.

In reinforcement learning, a **model** is a function that tells us how the environment will respond to our agent's actions. That is, given a state-action pair, the model should tell us about possible rewards and next-states.

Just as environments can be deterministic or stochastic, so can models. If we're modelling a stochastic environment, this means that each state-action can lead to multiple possible next-states and rewards, each with some probability of occurring. For example, imagine that we are the 'O' player in a game of tic-tac-toe, playing against an opponent who chooses its actions randomly. We decide to take the marked action in the state illustrated below. Depending on where our opponent moves next, we could end up in one of three different next-states.

**The example tic-tac-toe state-action pair that we are considering. The darkened symbol is the action being chosen by our agent this time-step.**

A **distribution model** tells us about all of the possible next-states and rewards, along with their associated probabilities of occurring. It is a perfect model of our agent's environment, 𝑝(𝑠′,𝑟|𝑠,𝑎), and is the type of model that we have previously seen DP methods assume access to.

**The information a distribution model would give us in response to this state-action pair. The model shows us every possible reward and next-state, along with their associated probabilities of occurring.**

For many problems, learning or explicitly defining a full distribution model is not feasible. An alternative is to use a **sample model**, which produces just one possible next-state and reward, sampled according to the probability of each possibility occurring.

**The information that a sample model might give us in response to this state-action pair. Instead of every possible outcome being given, only a single next-state and reward is sampled.**

We can use either type of model to **simulate** our agent's environment and generate **simulated experience**. Using a sample model, we could generate episodes of simulated experience one time-step at a time. Using a distribution model, we could theoretically generate all possible episodes, along with their probabilities of occurring.

# Planning in Reinforcement Learning

Completion requirements

In reinforcement learning, **planning** is any process that uses a model to improve an agent's policy. This definition might differ from those you have previously seen in other areas of AI.

Model−→−−−−−−PlanningPolicy

All of the planning methods we will introduce this week follow a common structure. First, we generate some simulated experience using a model of our agent's environment. This model might be provided, but will more often need to be learned by our agent through real interaction with its environment. Next, we use this simulated experience to update our agent's value estimates. Finally, our agent uses its updated value estimates to improve its policy, just as we have discussed previously.

Model⟶ Simulated Experience −→−−−−−−Updates Value Estimates ⟶Policy

Planning — improving an agent's policy using experience generated by a model — is often very similar or identical to learning — improving an agent's policy using real experience generated through interaction with the environment. We can use simulated experience as the basis of both policy evaluation and policy improvement, just as we can real experience.

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Random-Sample One-Step Q-Planning

Completion requirements

We are now ready to look at an incredibly simple planning method: **Random-Sample One-Step Q-Planning**. This planning algorithm is based on Q-Learning, and bears many similarities to it. The key difference is that it performs updates based on simulated experience generated by a sample model, rather than real experience from our agent interacting with the environment.

The pseudocode for this algorithm can be seen below.

Random-sample one-step tabular Q-planning

 Initialise, for all 𝑠∈𝑆,𝑎∈𝐴(𝑠) : 𝑄(𝑠,𝑎)∈ℝ (arbitrarily)  Loop forever:  Randomly select a state 𝑠∈𝑆 and an action 𝑎∈𝐴(𝑠) Send (𝑠,𝑎) to a sample model, obtain sample next-state 𝑠′ and reward 𝑟 Apply one-step tabular Q-Learning update: 𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼[𝑟+𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)]

**Pseudocode for the random-sample one-step Q-planning algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506226&forceview=1#footnote-1)

As you can see, this algorithm uses a sample model to generate single time steps of simulated experience, which our agent uses to improve its action-value estimates via Q-Learning updates.

In terms of convergence properties, this Q-Planning algorithm converges to the optimal policy for a given model under the same assumptions that Q-Learning converges for a given environment.

It is important to note that if our model does not faithfully represent the environment it is simulating, the optimal policy learned through planning will likely not be the optimal policy interacting with the real environment.

# Lesson 1 Summary

Completion requirements

We started this lesson by discussing an inefficiency of one-step TD learning methods, where it can take a while for value changes to propagate throughout the state-space. We also introduced **planning** methods, which use **models** to generate simulated experience which can be used to perform learning. We discussed

- how the updates made by one-step learning methods can be inefficient, in that a lot of training may be required for value changes to propagate across the state-space;
- how **models** can be used to generate **simulated experience** for use as part of a learning process;
- the difference between **distribution models** and **sample models**;
- that, in reinforcement learning, **planning** is any process that uses a model to improve an agent's policy; and
- how to implement the random-sample one-step Q-Planning algorithm.
# Lesson 2 Introduction

Completion requirements

In this lesson, we will look at how an agent can perform planning online while simultaneously acting and learning in its environment. We will see how online planning can help accelerate the propagation of value changes, addressing the inefficiency in one-step learning methods we identified at the start of the last lesson.

We will discuss how an agent can use real experience from interacting with its environment, not only to update its value estimates but also to learn a model of its environment. An agent can then use that model to generate simulated experience, with which it can further improve its value estimates through planning. We will finish by considering how to model an environment whose dynamics change over time.

# Integrating Acting, Planning, and Learning

Completion requirements

In the last lesson, we examined the Q-Planning algorithm that used simulated experience generated by a sample model to perform planning updates and improve an agent's value estimates. The question we will now consider is whether an agent can perform such planning updates online, alongside the usual updates it makes while generating real experience by interacting with its environment.

The answer is yes, and as we shall discuss, it actually makes a lot of sense to do so.

As our agent interacts with its environment, it generates real experience: taking an action 𝑎 in a state 𝑠 will lead our agent to some next-state 𝑠′ and earn it a reward 𝑟. In the context of model-based reinforcement learning, this real experience plays two important roles:

- It can be used to improve our agent's value estimates, as usual. We call this **direct reinforcement learning**, as our agent is learning directly from its experience.
- It can also be used to learn a model of our agent's environment. We call this **model learning**.

Once an agent has built a model of its environment through **model learning**, it can use this model to generate **simulated experience**. It can then use this simulated experience to further improve its value estimates by performing planning updates. Such updates are sometimes referred to as **indirect reinforcement learning**, because our agent's real experience is being used to indirectly perform updates via the model.

A diagram showing how an agent's real experience, model, value estimates, and policy interact is shown below.

**A diagram showing how an agent's experience can be used to perform direct reinforcement learning updates, as well as model learning. The learned model can then be used to perform additional planning, (or, indirect reinforcement learning) updates.**  [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506230&forceview=1#footnote-1)

Direct reinforcement learning methods are often simpler than indirect reinforcement learning methods and are not affected by the possibility of simulated experience generated by an incorrect model. However, indirect methods can use simulated experience to perform more updates and learn more from a limited set of real experience. In particular, planning updates can help value changes propagate throughout the state space more quickly, leading to indirect methods improving their policies faster than direct methods.

# The Dyna-Q Algorithm

Completion requirements

We are now going to take a look at **Dyna-Q,** a reinforcement learning algorithm that combines the direct reinforcement learning updates of Q-Learning with the planning/indirect reinforcement learning updates of random-sample one-step Q-Planning.

To facilitate these planning updates, it learns a deterministic model of our agent's environment. The full pseudocode for this algorithm is shown below.

Tabular Dyna-Q

Initialise, for all 𝑠∈𝑆,𝑎∈𝐴(𝑠) : 𝑄(𝑠,𝑎)∈ℝ arbitrarily  Model (𝑠,𝑎) arbitrarily Loop forever: 𝑆← current non-terminal state Choose 𝐴 from 𝑆 using 𝜖-greedy policy derived from 𝑄Take action 𝐴, observe next-state 𝑆′ and reward 𝑅 𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾max𝑎𝑄(𝑆′,𝑎)−𝑄(𝑆,𝐴)] Direct  R LModel (𝑆,𝐴)←𝑆′,𝑅 Model Learning Repeat 𝑛 times: 𝑆←random previously-visited state 𝐴←random action previously taken in 𝑆𝑆′,𝑅←Model(𝑆,𝐴) 𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾max𝑎𝑄(𝑆′,𝑎)−𝑄(𝑆,𝐴)] Indirect R L

**Pseudocode for the Dyna-Q algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506231&forceview=1#footnote-1)

Let's step through the code below.

Part 1 Direct reinforcement learning

As we enter the main loop of the Dyna-Q algorithm, we first encounter a section of pseudocode that looks remarkably like Q-Learning. Here, direct reinforcement learning updates are performed based on our agent's real experience. That is, after starting in a state 𝑆, taking an action 𝐴, and observing a next-state 𝑆′ and reward 𝑅, our agent improves its action-value estimates by performing a Q-Learning update, just as we have studied previously.

Part 2 Model Learning
After performing an update based on our agent's most recent experience, our agent then performs model learning. This version of Dyna-Q learns a deterministic model of the environment by recording the next-state 𝑆′ and reward 𝑅 it observes after taking an action 𝐴 in a state 𝑆.

Part 3 Indirect Reinforcement learning
The final section of this pseudocode is where indirect, planning updates are performed. It closely resembles the Q-Planning algorithm we discussed previously. Every direct reinforcement learning update is followed by 𝑛 planning (or, indirect reinforcement learning) updates. The simulated experience used to perform these planning updates is generated by randomly selecting previously visited state-action pairs, and querying our agent's learned model for the next-state and reward that follows them.

Incorporating indirect reinforcement learning updates allows our agent to perform more updates to its value estimates per time step of real experience gathered than if it relied solely on direct reinforcement learning methods. Performing these additional planning updates allows value changes to propagate much more quickly. Notice that, if we set 𝑛=0, the indirect reinforcement learning loop is never entered, and this algorithm reduces to regular Q-Learning.


### Stop and Think


##### The Dyna-Q algorithm learns a deterministic model of the environment by storing the latest next-state 𝑆′ and reward 𝑅 observed after executing an action 𝐴 in a state 𝑆. If our environment were stochastic, this model learning approach would not be suitable.

##### Have a think about how our agent could learn a model in a stochastic environment.

# Dyna-Q Performance

Completion requirements

We are now going to take a look at the results presented in the course textbook, which demonstrate that Dyna-Q learns more quickly than Q-Learning. We will then try to explain why this is the case by comparing the action-value updates performed by Dyna-Q and Q-Learning during training.

For this example, consider the gridworld illustrated below. The agent has access to four directional movement actions in each state, and attempting to move into an obstacle or leave the bounds of the gridworld causes the agent to stay in its current grid-square. The reward for all transitions is 0, apart from transitions to the goal state, where the reward is +1. This is a discounted episodic problem 𝛾=0.95.

**Dyna-Q gridworld example.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506232&forceview=1#footnote-1)

The results of training three Dyna-Q agents with different values of 𝑛 are shown in the graph below. Each agent started with all action-values estimates initialised to 0, followed an 𝜖-greedy policy with 𝜖=0.1, and used a learning rate 𝛼=0.1. Additional information about how these results were produced can be found in the course textbook.

Recall that, for the Dyna-Q algorithm, 𝑛 is the number of planning updates performed per time-step or real experience generated. So, an agent with 𝑛=50 will perform a 50 planning updates every time-step, whereas an agent with 𝑛=0 is equivalent to a Q-learning agent that performs no planning updates. Looking at the graph, it is clear that agents that performed more planning updates achieved much faster learning.

**Learning curves showing the performance of three different Dyna-Q agents, each with a different value of 𝑛, in our example gridworld.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506232&forceview=1#footnote-2)

Why is it the case that the planning agents learned so much more quickly than the agent which didn't perform any planning? Take a look at the figure below, which illustrates the action with the highest action-value in each state halfway through the second episode of training for the agents with 𝑛=0 and 𝑛=50. Grid squares with no arrows indicate grid squares where all action-values have not been updated, and are still zero.

**Illustration of the policies learned by two Dyna-Q agents after one and a half episodes of interaction in our example gridworld. The agent on the right performs 𝑛=50 planning updates per time-step, whereas the agent on the left does not perform any.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506232&forceview=1#footnote-3)

As we would expect after our discussion at the start of this week, the 𝑛=0 agent has updated only a single action-value, and changes to its value estimates have propagated back only a single step from the goal state. On the other hand, the indirect reinforcement learning updates performed at every time step by the 𝑛=50 agent have allowed value changes to propagate nearly all the way back from the goal state to the starting state. In the same number of time-steps that our 𝑛=0 agent has only updated a single action-value, our 𝑛=50 agent has already nearly learned the optimal policy! This is all thanks to its additional planning updates, which help propagate value changes throughout the state space more quickly.

# When the Model is Wrong

Completion requirements

So far, we have considered planning methods that start off with no prior knowledge about its environment and, through interaction with its environment, gradually constructs a model of the environment. Once an agent has recorded the next-state and reward that follows all possible state-action pairs, we might consider its model to be complete.

Unfortunately, learning a complete and perfect model of the environment is rarely achievable in practice. An agent's model might be incorrect for a number of reasons, for instance:

- The environment might be stochastic, and the agent might not have observed all possible outcomes for a given state-action pair.
- Our agent's environment might be non-stationary, and our agent might not have observed changes in its dynamics.
- Our agent's model might have been trained using function approximation, which may have led to imperfect generalisation.

In any case, errors in our agent's model are likely to lead it to learn a suboptimal policy after performing planning updates.

We will now consider two non-stationary gridworld environments, whose dynamics change over time. We will consider what happens when the agent's model becomes out-of-date, and then explore an extension to the Dyna-Q algorithm that encourages the agent to seek out long-unvisited state-action pairs to check whether its model is still accurate.

Consider the gridworld illustrated below. Our agent has access to four directional movement actions, which behave as usual. Rewards are set to 0 for all transitions except those leading to the goal state, which are set to +1; the agent must navigate around a wall to reach the goal state as quickly as possible. This is a discounted episodic task.

For the first 1000 time steps, the agent must navigate around the right-hand side of a wall of obstacles to reach the goal state. However, after the 1000 time-steps, the path on the right-hand side of the wall of obstacles is blocked off, and a new path opens up on the left-hand side.

**The blocked maze gridworld.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506233&forceview=1#footnote-1)

In this environment, our agent would initially learn about the short path to the goal around the right-hand side of the wall. Once this path is blocked, our agent's model will be incorrect, and it will need to learn to take the longer path around the left-hand side of the wall. Our agent is likely to discover and correct its modelling error quite quickly after time-step 1000, because its existing policy will try to take it along the shorter route which no longer exists.

Essentially, the problem has become harder after the change, and our agent will be drawn to exploit opportunities that no longer exist. In doing so, it will correct its model and value estimates and quickly converge upon the new solution.

Another example of a non-stationary environment is shown below. The one-step dynamics are the same as in the previous example, but the way the environment changes over time is slightly different. For the first 3000 time-steps, the agent must learn to take the long route around the left-hand side of the wall. After 3000 time-steps, a second, shorter path opens up around the right-hand side of the wall, but the existing path remains open.

**The shortcut maze gridworld.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506233&forceview=1#footnote-2)

Despite the new path being shorter than the previous one, a Dyna-Q agent will find it difficult to discover and exploit it. Early in training, our agent will already have modelled that there is no shortcut on the right-hand side of the wall. So, the value estimates it will have built up during training will not incentivise our agent to go out of its way to discover this new path. Even if our agent were following an 𝜖-greedy policy, it is very unlikely to explore far enough to discover the new shortcut once it opens up.

Interestingly, discovering and exploiting a better solution that becomes available partway through training has turned out to be a more insidious challenge than adapting to an already-known solution that becomes unavailable.

# The Dyna-Q+ Algorithm

Completion requirements

As we have discussed, there are many reasons our agent's model might be incorrect. In the gridworld examples we just studied, an agent's model may become out-of-date after the environment changes in some way.

Once again, we find ourselves delicately balancing exploiting what our agent already knows to maximise its return with exploring to ensure the solution it has found is indeed optimal. In the context of model-based reinforcement learning, this means exploring enough to ensure our model is as accurate as possible, while exploiting what it tells us to maximise our return.

We will now look at **Dyna-Q**+, a version of Dyna-Q that uses a simple heuristic to help ensure our agent's model stays up to date in non-stationary environments. It is based on the idea that the longer it has been since our agent last observed the outcome of a given state-action pair, the more likely it is that the dynamics of that state-action pair will have changed.

To encourage an agent to test state-action pairs it has not observed for a long time, it adds an **exploration bonus** to the reward for each state-action pair when performing planning updates. Upon randomly selecting a state 𝑠∈𝑆 and action 𝑎∈𝐴(𝑠), with model outputs 𝑠′ and 𝑟, Dyna-Q+ would perform the following planning update:

𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼[𝑟+𝑘𝜏𝑠,𝑎‾‾‾√+𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)]

Where 𝜏𝑠,𝑎 is the number of time-steps since the outcome of state-action pair 𝑠,𝑎 was last observed, and 𝑘 is a parameter of the algorithm which controls the emphasis placed on the exploration bonus. The full pseudocode for Dyna-Q+ is shown below.

**Note** that, aside from the added exploration bonus, it is identical to the pseudocode for Dyna-Q.

Tabular Dyna-Q+

 Initialise, for all 𝑠∈𝑆,𝑎∈𝐴(𝑠) : 𝑄(𝑠,𝑎)∈ℝ arbitrarily Model(𝑠,𝑎) arbitrarily  Loop forever:  𝑆← current non-terminal state  Choose 𝐴 from 𝑆 using 𝜖-greedy policy derived from 𝑄 Take action 𝐴, observe next-state 𝑆′ and reward 𝑅 𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾max𝑎𝑄(𝑆′,𝑎)−𝑄(𝑆,𝐴)] Direct R L Model (𝑆,𝐴)←𝑆′,𝑅 Model Learning  Repeat 𝑛 times: 𝑆← random previously-visited state 𝐴← random action previously taken in 𝑆𝑆′,𝑅← Model (𝑆,𝐴)𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝑘𝜏𝑠,𝑎‾‾‾√+𝛾max𝑎𝑄(𝑆′,𝑎)−𝑄(𝑆,𝐴)] Indirect R L

**Pseudocode for the Dyna-Q+ algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506234&forceview=1#footnote-1)

Dyna-Q+'s exploration bonus encourages an agent to go out of its way to try long-untested actions in states it has not visited for a long time. This additional exploration has its costs, but it is usually worthwhile in non-stationary environments because it helps our agent detect changes and keep its model up to date. We will now look at how Dyna-Q+ performs compared to Dyna-Q on the blocking maze and shortcut maze gridworlds we discussed earlier.

A cumulative reward graph illustrating the performance of a Dyna-Q agent and a Dyna-Q+ on the blocking maze gridworld is shown below.

**Learning curves showing the performance of a Dyna-Q agent and a Dyna-Q+ agent on the blocking maze gridworld.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506234&forceview=1#footnote-1)

As can be seen on the graph, the rewards earned by both agents plateau immediately after the environment is changed to block the route to the right of the wall. This is caused by both agents exploring below the barrier before finding a new route to the goal after their original route was blocked. After a short time, however, both agents found the new route and continued earning rewards.

Also noticeable is the fact that the Dyna-Q+ agent performs significantly better than the Dyna-Q agent during the first 1000 time-steps. This may be attributed to the Dyna-Q+ agent's more structured exploration compared to the Dyna-Q agent. This might have allowed the Dyna-Q+ agent to identify the true optimal route more quickly than the Dyna-Q agent, which might have taken a while to find it after first discovering a suboptimal route that reached the goal.

A cumulative reward graph comparing the performance of a Dyna-Q agent and a Dyna-Q+ agent in the shortcut maze gridworld is shown below.

**Learning curves showing the performance of a Dyna-Q agent and a Dyna-Q+ agent on the shortcut maze gridworld.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506234&forceview=1#footnote-2)

The difference in performance between the two agents after the change in dynamics is much more significant than in the previous example. We can see that the Dyna-Q+ agent discovered and began exploiting the shorter path quite soon after it became available, leading to higher reward rates. On the other hand, the Dyna-Q agent never found the shorter route and continued travelling along the left-hand side of the wall.

The Dyna-Q+ agent discovered the shortcut thanks to its exploration bonus. Eventually, the agent was encouraged to revisit the right-hand side of the wall to ensure its model of that part of the environment was up to date. The Dyna-Q agent lacked an exploration bonus to incentivise it to deviate from the route it had already discovered, so it never found the shorter route.

# Lesson 2 Summary

Completion requirements

In this lesson, we discussed how an agent's real experience can be used to construct a model of its environment, and how planning updates can be performed alongside direct reinforcement learning updates to improve the learning efficiency of our agents. We discussed

- how an agent can perform planning online, while interacting with its environment;
- how an agent can use real experience to perform **direct reinforcement learning** updates or **model learning**;
- how an agent uses its model to generate simulated experience, which can be used to perform planning or **indirect reinforcement learning** updates;
- how to implement the **Dyna-Q** algorithm;
- how **exploration bonuses**, such as that used by the Dyna-Q+ algorithm, can be used to help ensure that an agent keeps its model up-to-date; and
- how to implement the **Dyna-Q+** algorithm.
# Lesson 3 Introduction

Completion requirements

Previously, we have looked at Monte Carlo methods and one-step temporal-difference learning methods. As you will recall, Monte Carlo methods learn from full episodes of experience and do not bootstrap. On the other hand, one-step temporal-difference learning methods learn (as their name suggests) from single time steps of experience.

In this lesson, we will be introducing methods that bridge the gap between these two classes of solution methods. We call them 𝑛-step methods because they take into account information generated over 𝑛time-steps, bridging the gap between the one-step and Monte Carlo methods we have seen so far.

# n-Step Returns

Completion requirements

We can think of one-step temporal-difference learning methods and Monte Carlo methods as existing at opposite ends of the same spectrum. At one end of the spectrum, we look only one step into the future when performing our updates. At the other end, we look all up way up to the end of the episode. In this lesson, we will consider the temporal-difference learning methods that lie between these two extremes.

**Backup diagrams representing the updates performed by one-step TD and Monte Carlo prediction methods.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506240&forceview=1#footnote-1)

Consider the usual one-step return that we use as a target for one-step temporal-difference learning methods:

𝐺(1)𝑡=𝑅𝑡+1+𝛾𝑉(𝑆𝑡+1)

Bearing in mind the recursive relationship defined by the Bellman equations, we can decompose the value of the next-state in this one-step return further, and define a two-step return:

𝐺(2)𝑡=𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑉(𝑆𝑡+2)

Using this same idea, we can define an **𝑛-step return**, which takes into account 𝑛 future rewards and then bootstraps off of the next-state encountered after 𝑛 time-steps:

𝐺(𝑛)𝑡=𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑅𝑡+3+…+𝛾𝑛𝑉(𝑆𝑡+𝑛)

If we were to include all rewards up until the end of the episode at time-step 𝑇, we are left with the full return, as used in Monte Carlo updates:

𝐺𝑡=𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑅𝑡+3+…+𝛾𝑇−𝑡−1𝑅𝑇

**Backup diagrams for various 𝑛-step TD methods.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506240&forceview=1#footnote-2)

At one end of the spectrum, we have the one-step TD update, in the middle we have various 𝑛-step TD learning updates, and at the other end, we have the Monte Carlo update.

Being able to perform 𝑛-step TD updates is useful because they allow us to spread information back over more time steps than their one-step counterparts. Consider the simple 5×5 gridworld we discussed previously regarding the potential inefficiencies of one-step updates. Below is an illustration of the action-values two agents would have learned after one episode of interaction with this environment, with one agent performing one-step updates and the other performing four-step updates. Notice how the agent performing four-step updates has performed non-zero updates for four action-values, whereas the one-step agent has performed only one.

**An illustration of the differences between the action-value updates performed by one-step and four-step TD agents in the 5×5 gridworld after one episode.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506240&forceview=1#footnote-3)

The left panel shows the agents' trajectories. The centre panel shows the single action-value update performed by the one-step agent. The right panel shows the action-value updates performed by the four-step agent.

# Performance of n-Step Methods

Completion requirements

Implementing a temporal-difference learning method that performs 𝑛-step updates is relatively straightforward, but there are a few considerations we need to make.

Firstly, we must take into account the fact that our agent cannot actually look into the future when performing its 𝑛-step updates: as our update at time-step 𝑡 depends on the rewards 𝑅𝑡+1,𝑅𝑡+2,…,𝑅𝑡+𝑛, we have to wait until those rewards are observed before we can make any updates.

This also means that, secondly, we can't perform any updates for the first 𝑛−1 time-steps of each episode, and must perform 𝑛−1 updates after each episode ends.

Consider the environment illustrated below, made up of a chain of 7 states. The states on the far right and far left of the chain are terminal states. Our agent can deterministically move between adjacent states, and the reward for all transitions is 0 except for transitions to the far-right terminal state, which receive a reward of +1.

**An illustration of the random walk problem with seven states.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506241&forceview=1#footnote-1)

Let us now examine the performance of a temporal-difference learning prediction method similar to TD(0), but which uses 𝑛-step returns as its target instead of the one-step return. The environment being tested is a version of the random walk problem illustrated above with 19 states instead of 7, and the policy being evaluated is the equiprobable random policy

The graph below shows the average root-mean-squared error in the predicted values for each state for various values of 𝑛 and 𝛼 after 10 episodes of training.

**The performance of an 𝑛-step TD prediction algorithm on a 19 state random walk problem, for various values of 𝑛 and 𝛼.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506241&forceview=1#footnote-2)

As you can see, the trials run using intermediate values of 𝑛 managed to predict state values more accurately for a wide range of 𝛼 than trials run using 𝑛=1 (equivalent to one-step updates) and trials run using very high values of 𝑛. Indeed, it is often the case that intermediate 𝑛-step TD methods perform better than either one-step TD methods and Monte Carlo methods, but there is no single value of 𝑛 which is optimal for every problem.

# Complex Returns

Completion requirements

At this point, we have established that 𝑛-step TD methods can outperform one-step TD methods, and Monte Carlo methods that learn from full episodes. But, we are now left with a new problem: which value of 𝑛 should be choose for a given problem? As discussed previously, different values of 𝑛 may work better or worse for different problems.

As it turns out, we don't actually have to settle for a choosing a single value of 𝑛. In fact, we are free to update towards any weighted average of 𝑛-step returns.

We refer to updates towards a weighted sum of various 𝑛-step returns as **complex backups**.

For example, we could take the average of the two-step and three-step returns:

𝐺avg(2,3)𝑡=12+12[𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑉(𝑆𝑡+2)][𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑅𝑡+3+𝛾3𝑉(𝑆𝑡+3)]

**An illustration of the complex return composed of the two-step and three-step returns.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506242&forceview=1#footnote-1)

This is useful, because it means we can incorporate experience generated over multiple time scales into our updates. But, it doesn't completely address our original issue: we've simply shifted the problem from _"What value of 𝑛 should we use?"_ to _"What value**s** of 𝑛 should we use?"_

In the next lesson, we will develop a more satisfying solution to this problem that efficiently averages over all possible 𝑛-step returns, avoiding the need to select any specific value(s) at design time.

# Lesson 3 Summary

Completion requirements

In this lesson, we have introduced the concept of 𝑛-step methods, which bridge the gap between one-step TD methods and Monte Carlo methods. We have discussed

- how **𝑛-step methods** compute returns using rewards earned over 𝑛 time-steps,
- how 𝑛-step methods can lead to more efficient learning by allowing updates to convey information across multiple time steps,
- how different values of 𝑛 can lead to better or worse performance when applied to different problems, and
- how multiple different 𝑛-step returns can be combined to perform a **complex backup**.
# Lesson 4 Introduction

Completion requirements

In the previous lesson, we introduced the concept of 𝑛-step returns and discussed how 𝑛-step temporal-difference learning methods can help us learn more efficiently than the one-step and full-episode methods we have seen previously. We also discussed the difficulties in choosing which value(s) of 𝑛 to use for a given problem.

In this lesson, we will resolve this difficult by introducing the **𝜆-return**, which gives us a principled way to average all possible 𝑛-step returns. We will also introduce two concrete algorithms that use eligibility traces to perform updates towards the **𝜆-return**. As usual, we will start by looking at a **policy evaluation**or **prediction** algorithm for learning the value function for a given policy, before moving on to a **control algorithm** for learning the optimal policy for a given problem.

We will finish the lesson by comparing the updates performed by a one-step agent and various 𝑛-step agents in a gridworld to illustrate how 𝑛-step updates and updates toward the 𝜆-return can help our agent learn more efficiently.

# The λ-Return (Forward View)

Completion requirements

In this lesson, we will introduce the concept of the **𝜆-return**. The 𝜆-return is a particular weighted average of **all possible 𝑛-step returns**.

In other words, the 𝜆-return is a complex backup that incorporates all possible 𝑛-step backups, eliminating the need for us to choose specific value(s) of 𝑛.

The weight that the 𝜆-return applies to each 𝑛-step return is illustrated below.

**An illustration of the complex return composed of the two-step and three-step returns.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506248&forceview=1#footnote-2)

Here, we've introduced a new variable, 0≤𝜆≤1, that controls how much weight is placed on each 𝑛-step return. Returns that incorporate more distant future information are assigned less weight than those that incorporate more immediate information.

We can define the 𝜆-return formally as:

𝐺𝜆𝑇=(1−𝜆)∑∞𝑛=1𝜆𝑛−1𝐺(𝑛)𝑡

The one-step return is assigned the greatest weight of 1−𝜆, with the weight being decayed by a factor 𝜆 for every subsequent 𝑛-step return. After a terminal time-step 𝑇, all subsequent 𝑛-step returns will be equal to 𝐺𝑡, the full episode return. We can use this effect to split the definition of the 𝜆-return into two parts:

𝐺𝜆𝑇=(1−𝜆)∑𝑇−𝑡−1𝑛=1𝜆𝑛−1𝐺(𝑛)𝑡Until Termination+𝜆𝑇−𝑡−1𝐺𝑡After Termination

Writing out the 𝜆-return like this makes it easy to see the effect of varying 𝜆 to its extreme values of 0and 1. If we set 𝜆=0, we end up with one-step TD learning. On the other hand, if we set 𝜆=1, we end up with a Monte Carlo update that relies on only the full episodic sample return.

**TD(𝜆)** is a TD policy evaluation (or, prediction) algorithm that performs updates towards the 𝜆-return.

Computing the 𝜆-return using the equation above is referred to as the **forward view of TD(𝜆)**.

The _"forward view"_ refers to the fact that the equation above computes the 𝜆-return each time step based on information that only becomes available in the future. Notice that, because the 𝜆-return takes into account all possible 𝑛-step returns, we cannot actually compute it until the end of each episode. This leaves us in the same situation we found ourselves in when using Monte Carlo methods earlier in the unit, where our agent was unable to update its policy during each episode.

Because of this, computing the forward view of the 𝜆-return using the equation above is useful for theoretical analysis, but presents significant issues when it comes to implementation.

# Eligibility Traces (Backward View)

Completion requirements

Although we can compute the 𝜆-return using the equation we saw previously, the fact that our updates rely on future rewards poses significant implementation challenges. Specifically, it means we have to wait until the end of an episode before we can perform any updates, because the definition of the 𝜆-return incorporates the full episodic sample return as its final term.

Now, we will discuss the **backward view of TD(𝜆)**, which provides us with a convenient way of performing incremental updates towards the 𝜆-return every single time step, rather than having to wait and update all at once at the end of each episode.

In the backward view of TD(𝜆), we store an additional piece of information about each state, known as its **eligibility trace**. The eligibility trace for state 𝑆 at time-step 𝑡 is denoted 𝑒𝑡(𝑆). The eligibility trace for each state is initialised to zero, and is increased by +1 every time our agent visits a given state. After each time-step, the eligibility trace for each state is decayed by a factor 𝛾𝜆:

𝑒𝑡(𝑆)={𝛾𝜆𝑒𝑡−1(𝑆)+1𝛾𝜆𝑒𝑡−1(𝑆)if 𝑆=𝑆𝑡otherwise

where 𝛾 is the discount factor and 𝜆 is the variable we have previously associated with computing the 𝜆-return, which we will now refer to as the **trace-decay parameter**.

States that have been visited recently will have higher eligibility scores than those that have not been visited recently. Imagine that our agent visited a single state multiple times during an episode: the changes in the eligibility trace for this state are illustrated below.

**The eligibility trace for a single state which is visited multiple times over the course of an episode. The trace increases whenever the state is visited, after which it slowly decays.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506249&forceview=1#footnote-1)

The eligibility trace for a given state at a given time step indicates how **eligible** that state is to **receive credit** for any reward earned at that time step. Instead of updating only the value of the most recently visited 𝑛 states at each time step, as we would for an 𝑛-step TD method, **we update every state's value in proportion to its eligibility trace**.

Intuitively, eligibility traces let us spread information about a reward earned at a given time step back to all the states visited at previous time steps that contributed to our agent earning that reward. Recently visited states will have high eligibility traces, so they will be updated strongly. On the other hand, long-unvisited states will not be updated much, if at all.

As usual, the TD prediction error after observing the transition to a next-state 𝑆𝑡+1 and observing a reward 𝑅𝑡+1 by taking action 𝐴𝑡 in state 𝑆𝑡 will be:

𝛿←𝑅𝑡+1+𝛾𝑉(𝑆𝑡+1)−𝑉(𝑆𝑡)

After incrementing the eligibility of state 𝑆, we would perform the following update for all states:

𝑉(𝑆)←𝑉(𝑆)+𝛼𝛿𝑒(𝑆)

Using eligibility traces to perform these updates is referred to as the _"backward view"_ of TD(𝜆) because our agent sends information observed at the current time step back to states it has visited in the past, in proportion to each state's eligibility trace.

The updates we perform using eligibility traces as part of the backward view of TD(𝜆) are equivalent to the updates towards the 𝜆-return we discussed in the forward view. The full proof for this is beyond the scope of this course, but can be found in the first edition of the course textbook.

Notice that, once again, if we set 𝜆 to 0 or 1, we obtain one-step TD and Monte Carlo updates, respectively.

However, the Monte Carlo updates we can implement using eligibility traces and a 𝜆 value of 1 are an improvement over the ones we've seen previously. This is because we can perform these updates incrementally at each time step, rather than waiting until the end of the episode. Importantly, we can apply this type of Monte Carlo update even in a continuous setting, whereas previously we were limited to episodic problems.

# Prediction with the TD(λ) Algorithm

Completion requirements

We are now ready to look at the pseudocode for the TD(𝜆) algorithm. This is a TD prediction algorithm where updates towards the 𝜆-return are implemented using eligibility traces. At each time step, the value of every state is updated in proportion to its current eligibility trace. The full pseudocode for this algorithm is shown below.

TD (𝜆) for estimating  𝑉≈𝑣𝜋

 Input: a policy 𝜋 to be evaluated  Algorithm parameters: learning rate 𝛼, trace-decay parameter 𝜆 Initialise, for all 𝑠∈ : 𝑉(𝑠)∈ℝ arbitrarily 𝑒(𝑠)←0 Loop for each episode:  Initialise 𝑆 Loop for each step of episode: 𝐴← action chosen by 𝜋 in 𝑆 Take action 𝐴, observe reward 𝑅 and next-state 𝑆′𝛿←𝑅+𝛾𝑉(𝑆′)−𝑉(𝑆)𝑒(𝑆)←𝑒(𝑆)+1 For all 𝑠∈:𝑉(𝑠)←𝑉(𝑠)+𝛼𝛿𝑒(𝑠)𝑒(𝑠)←𝛾𝜆𝑒(𝑠)𝑆←𝑆′;

The pseudocode for the TD(𝜆) algorithm is very similar to its one-step variant, TD(0), which we have examined before. This is to be expected, because TD(0) is the algorithm we arrive at if we use 𝜆=0.

The pseudocode for this prediction algorithm starts as we would expect. It takes the policy we wish to evaluate as input and initialises an initial guess for the value of each state. We also initialise the eligibility trace for each state to 0.

The main loop of this algorithm also looks very familiar and follows the agent-environment interaction cycle we have come to expect. Our agent starts in an initial state, chooses and executes an action according to the policy we're evaluating, and observes a next-state and reward. We then compute the TD prediction error for the transition our agent has just observed, and increment the eligibility of the most recently visited state.

Instead of simply updating the value of the most recently-visited state, as TD(0) would do, TD(𝜆) performs an update for **every** eligible state. This spreads information about the experience generated at the current time step back to all previously visited states according to their eligibility. After performing updates for every state, the algorithm decays each state's eligibility by 𝛾𝜆.

# Control with the Sarsa(λ) Algorithm

Completion requirements

Now we've seen 𝜆-return methods used to solve the policy evaluation problem (i.e., learning the value function for a specific policy), we will now look at using them to solve the control problem (i.e., learning the optimal policy for a given MDP).

As usual, we can extend prediction methods to control by estimating action values instead of state values and then performing policy improvement. Applying this idea to TD(𝜆) allows us to define a version of Sarsa that updates towards the 𝜆-return.

Instead of storing eligibility traces for states, we will now store eligibility traces for state-action pairs. The update we perform to the eligibility trace for a given state-action pair 𝑆,𝐴 each time step becomes:

𝑒𝑡(𝑆,𝐴)={𝛾𝜆𝑒𝑡−1(𝑆,𝐴)+1𝛾𝜆𝑒𝑡−1(𝑆,𝐴)if 𝑆=𝑆𝑡 and 𝐴=𝐴𝑡otherwise

After computing the TD error 𝛿𝑡 given our agent's most recent experience, we perform the following update for every state-action pair each time step, according to its eligibility:

𝑄𝑡+1(𝑆,𝐴)←𝑄𝑡(𝑆,𝐴)+𝛼𝛿𝑡𝑒𝑡(𝑆,𝐴)

We call the version of Sarsa which uses eligibility traces to perform updates towards the 𝜆-return Sarsa(𝜆). The full pseudocode for the Sarsa(𝜆) algorithm can be seen below.

Sarsa(𝜆) for estimating 𝜋≈𝜋∗

 Algorithm parameters: learning rate 𝛼, trace-decay parameter 𝜆 Initialise, for all 𝑠∈,𝑎∈𝐴(𝑠) : 𝑄(𝑠,𝑎)∈ℝ arbitrarily 𝑒(𝑠,𝑎)←0 Loop for each episode:  Initialise 𝑆,𝐴 Loop for each step of episode:  Take action 𝐴, observe reward 𝑅 and next-state 𝑆′𝐴′← action chosen in state 𝑆′ by policy derived from 𝑄 (e.g. 𝜖-greedy) 𝛿←𝑅+𝛾𝑄(𝑆′,𝐴′)−𝑄(𝑆,𝐴)𝑒(𝑆,𝐴)←𝑒(𝑆,𝐴)+1 For all 𝑠∈,𝑎∈𝐴(𝑠):𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼𝛿𝑒(𝑠,𝑎)𝑒(𝑠,𝑎)←𝛾𝜆𝑒(𝑠,𝑎)𝑆←𝑆′;𝐴←𝐴′

**Pseudocode for the Sarsa(𝜆) algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1506251&forceview=1#footnote-1)

The figure below illustrates the differences in updates performed by one-step Sarsa, four-step Sarsa, and Sarsa(𝜆) agents after one episode of interaction along a fixed trajectory in the simple 5×5 gridworld we have used throughout this week to discuss the learning efficiency of various TD methods. Arrows in a grid square represent the action that would be taken by a greedy policy in that grid square with respect to our agent's action-value estimates at the end of the episode. Recall that, in this environment, all rewards are 0 except a positive reward for transitioning to the goal state (assume a discount factor 𝛾<1.0).

  
As we have discussed at length, our simple one-step Sarsa agent can only update the value of the action that it performed immediately prior to reaching the goal state during this first episode. It will take a lot of additional training to help these value changes propagate back to states near the start state.

Next, we can see the updates performed by a four-step method based on Sarsa. Information about the reward earned by our agent at the end of the episode is spread across four states, which clearly improves learning efficiency over the one-step method.

Finally, consider the updates performed by a Sarsa(𝜆) agent (for some intermediate value of 𝜆). According to the forward view, updating towards the 𝜆-return considers all possible 𝑛-step returns, so information about future rewards is incorporated into the update at every state. According to the backward view, our agent increments the eligibility trace for each action it executes, and then updates every eligible state-action pair when it eventually earns a reward at the end of the episode. Regardless of the view adopted, the Sarsa(𝜆) agent uses information about the reward it receives at the end of the episode to update the values of all of the actions it executed during the episode, resulting in value updates which propagate all the way back to the starting state.

# Lesson 4 Summary

Completion requirements

In this lesson, we introduced the **𝜆-return**, a specific way of averaging over all possible 𝑛-step returns. We discussed two equivalent views of the 𝜆-return: a **forward view,** which is useful for theoretical understanding, and a backward view, which uses **eligibility traces** to implement updates in practice.

We then examined two concrete algorithms that perform updates towards the 𝜆-return using eligibility traces. First, we looked at the TD(𝜆) algorithm for policy evaluation, and then looked at the Sarsa(𝜆) algorithm for on-policy control.

We have discussed

- how the **𝜆-return** is a specific way of averaging across all possible 𝑛-step returns,
- how to implement updates towards the 𝜆-return using **eligibility traces**,
- how to implement the **TD(𝜆)** algorithm,
- how to implement the **Sarsa(𝜆)** algorithm, and
- why 𝑛-step methods, and methods which perform updates towards the 𝜆-return, can learn more efficiently than one-step TD methods by spreading information back further to eligible states.

# Weekly Summary

Completion requirements

This week, we have looked at two ways we can improve the learning efficiency of our temporal-difference learning agents. We started by discussing how one-step TD methods are limited to propagating information about rewards only one time step at a time. By looking at an example gridworld where our agent only earns rewards at the end of an episode, we saw how this type of learning can be very inefficient and how information about rewards can take a long time to propagate across the set of states.

We introduced **planning methods** as a potential solution to this issue. We discussed how, in reinforcement learning, planning is any method that takes a **model** of our agent's environment as input and uses it to improve our agent's policy. We looked at planning methods like **Dyna-Q**, which learn a deterministic model of our agent's environment by recording the next-states and rewards that follow each state-action pair. We then perform additional updates using simulated experience generated by this model. These additional planning updates allow value changes to propagate throughout the state-space more quickly.

We then introduced **𝑛-step methods**, which bridge the gap between one-step TD methods and Monte Carlo methods by incorporating rewards earned over 𝑛 time-steps into each update. 𝑛-step methods allow information about rewards to propagate back to states visited 𝑛 time-steps in the past, enabling more efficient learning than one-step methods. However, 𝑛-step methods were not without their drawbacks, notably the need to wait 𝑛 time steps after visiting a state before performing any learning.

We then introduced the idea of **complex backups**, defined as a weighted average of various 𝑛-step returns. Along with this, we introduced the **𝜆-return**, a method for averaging all possible 𝑛-step returns. We discussed how updates to the 𝜆-return allow information about a reward to be applied to all recently visited states, and how adjusting 𝜆 allows us to shift between one-step TD and Monte Carlo-style updates.

Despite performing updates towards the 𝜆-return seeming tricky because we have to wait until the end of each episode before the final 𝑛-step return becomes available, we described how they can be implemented incrementally, per time step, using eligibility traces. We finished by looking at **TD(𝜆)** and **Sarsa(𝜆)**, a prediction method and a control method, respectively, that perform updates towards the 𝜆-return using eligibility traces.