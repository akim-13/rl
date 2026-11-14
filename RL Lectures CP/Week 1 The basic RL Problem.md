In this first week, we'll be starting from the very basics. We'll begin by characterising reinforcement learning methods and the problems they aim to solve, and discuss how they differ from the methods and problems encountered in other branches of machine learning. We'll then introduce some key reinforcement learning concepts more formally, and then finish the week by taking a (brief) look at our first concrete reinforcement learning algorithm: Q-Learning.

By the end of this week, you should be able to:

-     Describe the types of problems tackled in reinforcement learning, and explain how they differ from the problems tackled in other branches of machine learning.
-     Describe how a reinforcement learning agent interacts with its environment.
-     Describe, at a high level, how the Q-Learning algorithm works.

**Reinforcement learning diagram of a Markov decision process.**  [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490561#footnote-1)

In this lesson, we are going to introduce reinforcement learning (RL), the branch of machine learning we'll be focusing on in this unit. We will start off by trying to define what reinforcement learning is and what reinforcement learning methods aim to do. Then, we will move on to taking a look at a few high-profile examples of problems that reinforcement learning has been applied to in the past, before trying to characterise the class of problems that reinforcement learning methods aim to solve. 

In your previous units, you will likely have applied methods from other branches of machine learning, such as supervised learning, to various types of problems, such as classification and regression problems. As we will see in this lesson, reinforcement learning methods aim to tackle sequential decision problems, which are fundamentally different to the types of problems targeted by methods from other branches of machine learning.


Being able to learn through interaction with one's environment is a key hallmark of intelligence in humans and animals. Think about how you make decisions and learn in your everyday life: based on your current situation, you choose to act in some way; based upon your actions, the situation changes; based on how good (or bad) those changes were for you, you might decide to act differently if you encounter the original situation again in the future.

In brief, reinforcement learning (RL) is a computational approach to goal-directed learning through interaction.

Let's explore this below.


We refer to this as the **agent-environment interaction** cycle[[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490562&forceview=1#footnote-1).

- **Environment and agent:** In the reinforcement learning framework, we consider an agent situated in an environment with which it interacts.
- **Action:** Our agent interacts with its environment by taking actions. These actions will cause the environment to change in some way. Note that our environment might be stochastic, so taking the same action in the same state might have different consequences from one attempt to the next.
- **State:** Our agent is able to observe the immediate consequences of its action. That is, the agent will observe that the environment started off in one state, and then transitioned to some new state as a result of its action. This cycle of interaction, with the agent going from state to next-state by choosing actions continues over and over again.
- **Reward:** Our agent doesn't want to just be taking a random walk through its environment — its behaviour will be directed by some goal. The way we represent an agent's goals in reinforcement learning is through the reward signal. This means that after taking an action and observing the new state, our agent also receives some numerical reward for its action.
- **Maximise reward:** In order to achieve its goals, then, our agent's objective is to choose actions that maximise the total reward that it is able to earn over the long term.

Reinforcement learning can be thought of as learning how to act — learning how to map situations to actions — in a way that lets you accumulate as much reward as possible in the long term.

The agent-environment interaction cycle is a key element of reinforcement learning, and we will see it appear in some form in all the problems and learning algorithms we encounter in this unit. We will be seeing many variations of the above diagram as we progress through the unit, as we start to add more detail to the concepts we've just introduced.

# What can reinforcement learning do?

Completion requirements

Now that we have introduced reinforcement learning as a computational approach to goal-directed learning from interaction, we can now start to look at the type of problem that reinforcement learning can be used to solve.

If you have ever heard of reinforcement learning prior to starting this unit, it is likely that you will have come across one or more high-profile results from research groups like DeepMind and OpenAI. To give you an idea of the type of problems that reinforcement learning methods aim to solve, we will start off by looking at some of these high-profile results.

**Notice** how, in each of these high-profile problems where reinforcement learning has been applied, we can quite easily see the agent-environment interaction cycle in action. In each case, we have an agent learning how to take a sequence of actions in order to accomplish some goal. The nature of the state that the agent observes, of the actions that it takes, and of the rewards it receives will be different for each of these problems, but they are nonetheless present.

**AlphaGo agent playing against Lee Sedol.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-1)

In perhaps one of the most well-known reinforcement learning results in recent years, DeepMind trained their AlphaGo Zero agent to play Go, a very popular deterministic two-player board game in many parts of the world. Starting knowing nothing about the game of Go, and by playing against itself many times, AlphaGo Zero [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-2) used reinforcement learning methods to learn how to play Go so well that it was able to outperform the human world champion, Lee Sedol.

This was a major achievement, because Go was much more difficult than other games that AI had managed to attain superhuman performance in, like chess. It wasn't expected that AI would be able to attain such high levels of performance in Go so soon.

Other versions of the AlphaGo Zero agent, and its relations, were developed which were able to play other games, like chess and shogi from scratch through self-play.


**A backgammon board.** [[3]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-3)

DeepMind's AlphaGo was by no means the first success that reinforcement learning methods had achieved in the domain of board games. One of the first high-profile applications of reinforcement learning methods was Gerald Tesauro's TD-Gammon agent in 1992 [[4]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-4)

Like AlphaGo Zero, TD-Gammon started off with no knowledge about how to play backgammon, but eventually managed to reach superhuman levels of performance on the game through self-play, and arguably became the best player in the world at the time. Backgammon is a much less difficult game than Go, but TD-Gammon was still a major early achievement for reinforcement learning methods. Of note is that unlike Go, which is entirely deterministic, Backgammon features stochasticity in the form of dice-rolls. Generally, reinforcement learning methods can be applied to both deterministic and stochastic problems.


**DeepMind's DQN agent learning to play Breakout.** [[5]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-5)

Of course, reinforcement learning methods haven't only been applied to board games. One of DeepMind's other recent high-profile successes was when they managed to train reinforcement learning agents to play Atari arcade games, like Space Invaders and Breakout, completely from scratch using pixel inputs. Their DQN [[6]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-6) agent managed to achieve superhuman results on many different arcade games and kickstarted a wave of research in the field now known as 'Deep reinforcement learning'.

This field exists at the intersection of deep learning and reinforcement learning, and we will examine it in detail towards the end of the unit.


**A robotic quadrupedal agent walking, before and after training.** [[7]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-7)

Finally, we'll look at an application of reinforcement learning algorithms in the physical world. For many years, annual 'RoboCup' competitions have been held where teams of intelligent robots play football against each other. Many machine learning and computer vision methods, including reinforcement learning methods, have been applied over the years in this competition.

The videos shown above show robotic agents from a team who entered the RoboCup competition in 2004 [[8]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490563&forceview=1#footnote-8). Reinforcement learning algorithms were used to train the robotic agents to walk forward as quickly as possible. In the first video, you can see the agent's initial walking strategy, learned using basic evolutionary methods. In the second video, you can see their agent's final walking strategy, learned using reinforcement learning methods while letting the agent walk across the RoboCup field for three hours.

It is clear that the walking strategy arrived at by the reinforcement learning methods allowed the agent to move much more quickly than its initial walking strategy.


We refer to problems like this, where an agent must make a sequence of decisions in order to accomplish some goal, as **sequential decision problems**.

This is the class of problems that reinforcement learning methods aim to solve. Later, we will learn how to formalise sequential decision problems as **Markov Decision Processes (MDPs)**.

The examples shown here are just a handful of high-profile sequential decision problems that reinforcement learning methods have managed to solve. However, there are many, many other application areas for reinforcement learning. Essentially, if a problem requires an agent to make a sequence of decisions in order to achieve some goal in its environment, then reinforcement learning methods can likely be of some use.

Try to **think** of three more problems that fit the description of a sequential decision problem.

Many games naturally fit this description, but can you think of any problems that aren't games?

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Key Features of Reinforcement Learning

Completion requirements

To finish this first lesson, we are going to tie together the key features of reinforcement learning that we have introduced so far, and then consider how reinforcement learning differs from supervised learning and unsupervised learning.

So far, we have said that reinforcement learning is a computational approach to learning from interaction, and that it is goal-directed. That is, the objective of a reinforcement learning agent is to learn what to do — how to map situations to actions — in order to achieve its goals.

We represent our agent's goals through a numerical reward signal, which our agent receives each time it takes an action that causes its environment to change. In order to achieve its goals, our agent must learn to maximise its long-term reward. It might seem like having a single numerical reward to guide our agent is too simple to represent challenging goals in complex environments. However, as we shall see during this unit, this notion of a goal is surprisingly flexible and usually sufficient to represent many complex goals in many different environments.

Finally, we discussed that the type of problems we aim to tackle using reinforcement learning methods are sequential decision problems: problems where our agent must make a sequence of decisions in its environment in order to achieve some goal.

### RL as a Branch of Machine Learning

It might be tempting to view reinforcement learning as part of supervised learning, because mapping situations to actions might look a bit like a classification problem. However, this is not the case: our agent works completely unsupervised, and is never told the optimal action to choose in any situation. Instead of relying on a supervisor to tell it what action to take during learning, our agent learns via trial-and-error interaction with its environment. It starts by exploring its environment randomly, and uses the rewards it earns in order to inform its future action choices.

A recurring theme in reinforcement learning is that of the trade-off between exploration and exploitation:

- When should an agent **explore** by choosing suboptimal actions in order to find potentially better ways of acting?

- When should our agent **exploit** what it already knows in order to earn as much reward as possible?

This trade-off is a fundamental part of reinforcement learning, and we will see several ways of dealing with it throughout this unit.

Since our agent does not rely on any external supervisor to tell it which actions are optimal during learning, it might be tempting to view reinforcement learning as part of unsupervised learning. However, this is not quite right either: the goal of unsupervised learning is to find hidden structures in data, whereas our goal in reinforcement learning is to map situations to actions in a way that lets us maximise some reward signal.

Given all of this, **it is correct to consider reinforcement learning as its own branch of machine learning**, separate from supervised learning and unsupervised learning, with its own class of problems and solution methods.


![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Lesson 1 Summary

Completion requirements

In this lesson, we introduced you to the exciting and fast-moving field of reinforcement learning. We have

- define **reinforcement learning as a computational approach to goal-directed learning from interaction**,
- introduced the **a****gent-environment interaction cycle**, where an agent observes its environment, takes an action that changes the environment's state, and earns a reward,
- introduced the class of problems that reinforcement learning methods aim to solve, known as **sequential decision problems**,
- noted a few high-profile examples of sequential decision problems that have been tackled using reinforcement learning methods, and
- considered how **reinforcement learning is a branch of machine learning in its own right**, separate from supervised learning and unsupervised learning, with its own class of problems and solution methods.

# Lesson 2 Introduction

Completion requirements

In the previous lesson, we started looking at reinforcement learning and characterised it as a computational approach to goal-directed learning through interaction. We introduced the agent-environment interaction cycle, which lies at the heart of reinforcement learning: our agent observes its environment, takes actions that change its environment, and earns rewards. In this lesson, we will add more detail to this agent-environment interaction cycle and begin to formalise many of the concepts we introduced at a high level in the last lesson.

By the end of this lesson, we will have all the necessary information to describe how to approach solving a simple sequential decision problem using reinforcement learning methods.

# The Agent-Environment Interaction

Completion requirements

Below is a slightly more detailed version of the figure that we started looking at towards the beginning of last lesson, which described the agent-environment cycle. Many of the components of this diagram should be familiar to you from last lesson — we have spoken about things like agents, environments, states, actions, and rewards rather loosely in the past, and we will now begin to formalise them.


**The agent-environment interaction.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1490569&forceview=1#footnote-1)

The agent is our learner in reinforcement learning, our decision-maker who chooses which actions to take. Our environment, then, is everything outside of our agent's direct control, which is changed by and responds to our agent's actions. Sometimes this divide between agent and environment is obvious, but sometimes it can be tricky to define; in the game of chess, for instance, we consider our agent's opponent to be part of the environment.

As we have discussed previously, the agent and environment interact continually. After observing the environment's current state, our agent selects an action. The environment responds to this action by changing to some new state,  and our agent earns some reward.

We'll begin formalising this cycle of interaction and introduce some notation.

In this course, we will consider the interaction between an agent and its environment to happen at discrete **time-steps,** 𝑡=0,1,2,3,…. At each time-step 𝑡, our agent observes some representation of the current **state** of the environment, 𝑆𝑡, and chooses an **action**, 𝐴𝑡.

One time-step later, 𝑡+1, the environment transitions to a **new state,** 𝑆𝑡+1, and the agent receives some numerical **reward,** 𝑅𝑡+1.

This repeats over and over again, giving a rise to a sequence of states, actions, and rewards, as shown in the figure below.

**Three time-steps of interaction between an agent and its environment.**

We refer to the sequence of states, actions and rewards generated as an agent interacts with its environment as a **trajectory**, denoted 𝜏.

The trajectory generated by the interaction in the above figure would start as 𝜏=𝑆0,𝐴0,𝑅1,𝑆1,𝐴1,𝑅2,𝑆2,𝐴2,𝑅3,𝑆3,…. This trajectory of experiences, the history of all interactions between an agent and its environment, is the basis of all of our agent's learning in reinforcement learning.

The interaction between an agent and an environment could continue indefinitely, or it might stop once the agent reaches a certain state. We will discuss this issue in more detail next lesson.

# The Expected Return

Completion requirements

Previously, we have discussed the concepts of rewards and goals rather loosely. We have said that we specify an agent's goals in terms of the reward signal, which the agent receives after taking each action, and that its objective is to maximise the reward that it earns over the long term.

How can we formalise this objective? What is the mathematical function that we are trying to maximise?

When we say that we want our agent to earn as much reward as it can in the long term, what we really mean is that we want our agent to maximise some function of 𝑅1,𝑅2,𝑅3,…. What function should we use? There have been various suggestions for useful objective functions in the literature, but the most commonly used is the **discounted return**. This is the function that we will be choosing to maximise in this unit.

We define the discounted return at time 𝑡, 𝐺𝑡, as:

𝐺𝑡≐𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑅𝑡+3+… 𝐺𝑡=∑𝑘=0∞𝛾𝑘𝑅𝑡+𝑘+1for 0≤𝛾≤1

In the case where 𝛾=1, 𝐺𝑡 is simply the sum of all rewards our agent earns after the time-step 𝑡. It makes perfect sense to use this as an objective function, since we want our agent to maximise the sum of rewards it receives in the future.

What is 𝛾, and what does it mean when 𝛾<1?

We refer to 𝛾 as the **discount factor**, and it controls how much our agent favours long-term rewards over short-term rewards. Low values of 𝛾, closer to 0, make our agent more short-sighted, and favour maximising short-term rewards instead of long-term rewards. A reward received 𝑘 time-steps in the future is only worth 𝛾𝑘−1 times as much as it would be if it was received immediately.

If the discount factor, 𝛾, is 0, then the agent will only ever care about its next reward — all other future rewards will be ignored. On the other end of the scale, if we set the discount factor to 1, our agent will value all future rewards equally — the reward it gets after its next action will be valued the same amount as the reward it earns in 100 time-steps' time.

We can relate the idea of discounting future rewards to how we make decisions in our daily lives. If we know something bad might happen tomorrow, we will probably do everything we can to avoid it. However, if we know that the same bad thing might happen in two years' time, we might well put off doing anything about it until much later. In other words, we will have discounted that future negative reward due to how far in the future we will be receiving it.

The other reason that we need to include the discount factor is, essentially, as a mathematical necessity. Because the total discounted return is a summation over all future rewards, it is an infinite sum that will diverge. Using a discount factor 𝛾 causes infinitely distant rewards to be multiplied by zero, stopping this infinite sum from diverging.

So, although there exists a nice, intuitive explanation for why a discount factor 𝛾<1 is useful, we also need it to ensure that our infinite sum does not diverge.

Something that is important to note, and which will become significant later in the unit, is that the returns at successive time-steps are related to each other, and can be defined recursively:

𝐺𝑡≐𝑅𝑡+1+𝛾𝑅𝑡+2+𝛾2𝑅𝑡+3+… 𝐺𝑡=𝑅𝑡+1+𝛾(𝑅𝑡+2+𝛾𝑅𝑡+3+…) 𝐺𝑡=𝑅𝑡+1+𝛾𝐺𝑡+1

# Learning a Policy

Completion requirements

Now that we have formalised the interaction of our agent with its environment and defined a function that our agent should aim to maximise, we are almost ready to start thinking about how we can apply reinforcement learning ideas to solving a very simple problem.

Before we do that, though, we have one last question to answer: **what** specifically should our agent be learning?

We have previously described reinforcement learning as learning how to act — how to map states to actions — in a way that maximises the discounted return. This, in fact, is the answer to our question:

**Our agent should learn how to act — a mapping from states to actions.**

We call such mappings a **policy**, and they are one of the most fundamental concepts in reinforcement learning.

Formally, a policy is a mapping from state-action pairs to the probability of taking a given action in a given state. We usually denote an agent's policy by 𝜋.

So, 𝜋𝑡(𝑠,𝑎) gives us the probability of our agent choosing action 𝑎 in state 𝑠 at time 𝑡.

You may see policy functions written as 𝜋(𝑎|𝑠), usually to remind you that this is a conditional probability distribution (i.e., the probability that the agent selects action 𝑎 **given** that it is in state 𝑠).

Notice that this definition of the policy gives us the freedom to represent both 

- **deterministic** policies, where the probability of choosing one action is 1.0 while the probability of choosing any of the others is 0.0, and
- **stochastic** policies, where our agent might choose between several different actions with different probabilities.

Now that we have introduced the concept of a policy, we are ready to say what a reinforcement learning method does.

A reinforcement-learning algorithm updates the agent’s policy in response to the trajectories it experiences, answering the question: "How should the agent adjust its behaviour after interacting with the environment?"

Our agent will begin with some initial policy — perhaps acting randomly, assigning equal probabilities to every action — and, as it generates experience, the reinforcement learning algorithm will improve the policy in a way that increases the discounted sum of future rewards the agent will expect to earn.


# Solving a Simple Problem Using Reinforcement Learning

Completion requirements

To tie together all of the concepts we have introduced and formalised this lesson, we are going to apply a reinforcement learning algorithm to solving a simple problem.

The specific algorithm we'll be applying is called **Q-Learning**, and it's one of the most fundamental and important algorithms in reinforcement learning. It will take a few more weeks before we can fully understand all the details of exactly how it works, but we already have everything we need to see it in action and think about how it works at a high level.

Seeing this algorithm now should give you a good idea of the types of methods we'll be building towards over the next few weeks.

#### **The Gold Gridworld Problem**

---

Let us start by defining a simple problem to solve using Q-Learning. The example we'll be using is called the Gold Gridworld problem, illustrated below. This problem involves a simple three-by-three grid containing a gold bar and a blue circle representing our agent.


**The Gold Gridworld.**

Each time-step, our agent can choose to move north, south, east, or west one grid square. However, these actions are unreliable - our agent's action succeeds as intended with probability 0.8, but is replaced by a randomly chosen action (including the intended action) with probability 0.2. If an action causes our agent to move into a wall, time advances but our agent remains in its current grid square. Each time-step, the agent is able to sense which grid square it is in, but cannot see beyond its current grid square.

The gold gridworld is a simple game where our agent must travel from its starting grid square to the grid square containing the gold as quickly as possible. Once the agent arrives at the gold, the game ends.


# Lesson 2 Summary

Completion requirements

In this lesson, we formalised many concepts related to the agent-environment interaction, and looked at applying reinforcement learning ideas to solving a simple sequential decision problem. We have

- formalised several key components of the agent-environment interaction cycle. Namely **states**, **actions**, **rewards**, **time-steps**, and **trajectories**,
- formalised the idea of training an agent to maximise its long-term reward by introducing the **total discounted return**,
- introduced the concept of **a** **policy**, a function that maps states and actions to probabilities,
- discussed that the goal of a reinforcement learning algorithm is to modify the agent's policy in a way that allows it to maximise the total discounted return, and
- applied the key reinforcement learning ideas introduced so far to the simple 'Gold Gridworld' problem.

# Weekly Summary

Completion requirements

This week, we've taken our first high-level steps in the field of reinforcement learning and introduced many of the core concepts underpinning it.

We began by taking a high-level look at how a reinforcement learning agent interacts with its environment. We then introduced the type of problems typically solved in reinforcement learning — sequential decision problems — and discussed how these problems are fundamentally different from the types of problems solved in other branches of machine learning.

We then added some detail to our initial, very high-level overview of reinforcement learning. We formalised many of the concepts related to the agent-environment interaction cycle, and ended up surmising that the goal of a reinforcement learning algorithm is to update our agent's policy such that it can earn as much reward as possible in the long term, an objective we formalised as the total discounted return.

Finally, we examined a concrete reinforcement learning algorithm — Q-Learning — and studied how it used an agent's experience to learn a policy that maximises the expected return.