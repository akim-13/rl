This week, we will begin adding more detail to the high-level picture of reinforcement learning that we started developing last week. We'll start by defining sequential decision problems formally as Markov Decision Processes (MDPs), and then look at a family of methods for solving them (i.e., finding optimal policies). Specifically, we will be taking a look at Dynamic Programming methods.

By the end of this week, you should be able to:

- Define a sequential decision problem as a Markov Decision Process.
- Explain what value functions are, and why they are useful.
- Explain the difference between state-vale functions 𝑣𝜋(𝑠) and action-value functions 𝑞𝜋(𝑠,𝑎).
- Define the Bellman Optimality Equations for 𝑣∗ and 𝑞∗.
- Explain how Dynamic Programming methods can be used to solve simple sequential decision problems whose dynamics are fully known.
- Implement the Value Iteration algorithm.

# Lesson 1 Introduction

Completion requirements

In this lesson, we will continue to add detail to key concepts in reinforcement learning that we have discussed so far in a broad sense. We will introduce two types of sequential decision problems — episodic and continuing tasks — and a unified notation that allows us to cover both cases. We will then formalise our definition of sequential decision problems by introducing Markov Decision Processes (MDPs).

MDPs give us a simple and flexible way of formally describing sequential decision problems. Despite their simple formulation, we will find that they can be used to model surprisingly complex sequential decision problems; it is possible to formulate all the problems we'll see in the first half of this unit as MDPs.

# Episodic and Continuing Tasks

Completion requirements

In the last lesson, we discussed how the agent-environment interaction cycle proceeds step-by-step at discrete time steps.

**Three time-steps of interaction between an agent and its environment.**

We will now discuss two different types of sequential decision problems that proceed in this step-by-step manner, but differ in how the interaction terminates. These two types of tasks are known as **episodic tasks** and **continuing tasks**.

The first type of problem we'll consider is **episodic** tasks. In this type of problem, an agent's interaction with its environment breaks naturally into a number of separate, independent **episodes**.

There are many examples of problems like this: learning to play chess can be broken down into individual games, and learning to reach the gold in the gold gridworld problem from the previous lesson can be broken down into individual journeys from the starting grid square to the gold grid square. Essentially, any problem where there is a pattern of repeated interaction with the environment which has a clear start and end point.

An episode ends when the agent finds itself in a special type of state known as a **terminal state**. This is followed by a reset, where the environment returns to some initial state, and a new episode of interaction commences.

There could be a single starting state that is reverted to after each reset, or one could be picked based on some distribution over a number of possible starting states. The initial state of each episode is chosen completely independently of what happened during the previous episode; whether or not our agent won its previous game of chess has no impact on the arrangement of pieces at the start of the next game. Of course, our agent can use what it learned in the previous episode to inform its action choices in the upcoming episode, but its actions in the previous episode have no impact on the environment after the reset.

Our agent's interaction with its environment will start at time-step 0 each episode, and will finish at some time-step 𝑇, where 𝑇 is a random variable which will vary from episode to episode. Some episodes might last 10 time-steps, some might last 1000, depending on things like our agent's chosen actions and stochasticity in the environment.

The second type of problem that we'll consider is **continuing** tasks. Unlike episodic tasks, continuing tasks **continue indefinitely**. There are no terminal states, and no complete resets of the environment to some starting state.

Consider a Mars rover with a long lifespan learning to navigate its surroundings and collect soil samples - there is no natural break in this agent's interaction with its environment, nor is there any opportunity to reset the state of the environment, so it cannot be cleanly divided into episodes.

Unlike the episodic case, where each episode will consist of a finite number of 𝑇 time-steps, interaction in the continuing case starts at time-step 0 and continues indefinitely. This can be potentially problematic for our definition of the return, which our agent is trying to maximise, since it is now an infinite sum. Because of this, choosing a discount factor 0≤𝛾<1 becomes important, as the discount term will tend towards 0 for increasingly distant rewards.

# Example: The Mountain Car Problem

Completion requirements

To illustrate the differences between episodic and continuing tasks, we'll examine a simple problem from the literature and define it as both an episodic task and a continuing task. The specific problem we'll be looking at is the Mountain Car problem. This is a classic toy problem commonly used in the reinforcement learning literature, which we will revisit throughout the unit.

The problem begins with a car positioned in a valley between two hills, and its objective is to drive up to the top of the hill to its right as quickly as possible. However, the car is underpowered and cannot reach the top of the right hill using its engine power alone. To reach the top of the right hill, it first needs to reverse partway up the left hill (away from the goal) and then drive forward towards the right hill. Using both its engine power and the momentum that it gains by travelling down the left hill, it will be able to overcome gravity and reach the top of the right hill.


**A diagram illustrating the Mountain Car problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496505&forceview=1#footnote-1)

There is a clear opportunity to treat this problem as an episodic task. We can treat the goal states, where the agent is at the top of the right hill, as terminal states. Then, moving the car back to its initial position at the bottom of the hills is clearly our 'reset'. In terms of rewards, we can penalise our agent −1 per time step to encourage it to reach the goal as quickly as possible.

However, despite this clear opportunity to break this interaction into episodes, we could also treat it as a continuing task. We will still need to allow our agent's position to be reset whenever it reaches the goal, but we will simply treat the interaction as continuing, rather than starting a new episode. In terms of rewards, we can reward our agent +1 whenever it reaches its goal, and apply discounting. The return would then be related to 𝛾𝐾−1, where 𝐾 is the number of time-steps that elapse before our agent reaches its goal, so our agent will be encouraged to reach the goal as quickly as possible.

In both cases, the return would be maximised by our agent trying to reach the top of the right hill as quickly as possible. Hopefully, this illustrates how we might treat a problem as both an episodic task and a continuing task. Most of the problems we'll see in this unit will be episodic problems, but it is important to be aware of the continuing case, too.

# A Unified Framework

Completion requirements

It is useful to be able to distinguish between these episodic and continuing sequential decision problems. However, having this separation does leave us in a  troublesome position where we might have to treat problems differently, at least mathematically, based on whether they are episodic or continuing.

To avoid worrying about this, we will now introduce a unified framework for working with both types of problems. The main difference we need to consider is that we define the return as a sum over a finite number of time-steps in the episodic case, whereas we define it as a sum over an infinite number of time-steps in the continuing case.

How can we unify the mathematics used in these two cases?

Luckily for us, it can be done quite simply. We will do it by treating episode termination as a transition to a special **absorbing** state. This state only allows transitions to itself, and these transitions give rewards of 0. Essentially, once our agent reaches a terminal state, it can never escape or earn any future rewards. This is illustrated in the figure below.

**An example of the unified notation, with an episode of interactions ending in an absorbing state.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496506&forceview=1#footnote-1)

Take the episode of interaction shown in the above figure as an example: we get the same return whether we sum only over the first 𝑇 time-steps (𝑇=3), as in the episodic case, or over the full infinite sequence. This is because all interactions beyond time-step 𝑇, after which the agent is stuck in the absorbing state, produce a reward of zero.

We can now cover both the episodic and continuing case using the same mathematical tools, including possibilities where 𝑇=∞ or 𝛾=1 (but not both simultaneously!).

# Markov Decision Processes (MDPs)

Completion requirements

Now we've established that interaction between the agent and its environment can unfold in one of two ways - either terminating at some terminal time step or continuing indefinitely - and that we have come up with a unified notation that elegantly handles both cases, we are now in a position to define sequential decision problems formally, as some kind of mathematical object.

Specifically, we will formulate sequential decision problems as **Markov Decision Processes** (**MDPs**).

### What is a State?

Before we can introduce Markov Decision Processes, we need to provide a bit more detail on what we mean by the term **state**.

So far, we've said that the state at time-step 𝑡 consists of whatever information is currently available to our agent about its environment. In practice, this information could be made up of many things: it could be immediate sensations, it could be highly processed sensations, it could be information collected by the agent over time from previous sensations. It could be very simple, such as a single number representing which grid square our agent is currently in, or more complex, such as pixel inputs from a camera.

### Markov Representation of State

What's important, though, is what we want this information to convey. Ideally, we want the current state to summarise everything that has previously happened in the environment that is relevant to our agent's decision-making process. More formally, we say our state representation should have the **Markov property**.

This means that if we are given the state at time-step 𝑡, 𝑆𝑡, and the selected action, 𝐴𝑡, having access to the rest of our agent's history (i.e., previously visited states, previously selected actions, and previously received rewards) would yield no additional information. Formally:

𝑝(𝑆𝑡+1,𝑅𝑡+1∣𝑆0,𝐴0,𝑅1,…,𝑅𝑡,𝑆𝑡,𝐴𝑡)=𝑝(𝑆𝑡+1,𝑅𝑡+1∣𝑆𝑡,𝐴𝑡)

Working with problems where this Markov property holds is useful, because it means that our agent only needs to keep track of what has just happened, and can happily forget about the rest of its history. This avoids our agents having to keep track of an ever-increasing list of previous experiences, because they can all be summarised by the most recent experience.

Consider the game of tic-tac-toe, where our agent is the O player in the state illustrated below. There are multiple ways that our agent could have ended up in this state: the X player might have chosen the top-right cell first, or the bottom-left cell first.

![An example 3x3 game grid of tic-tac-toe with a circle in the middle cell and crosses in the bottom-left and top-right cells.](https://moodle.bath.ac.uk/pluginfile.php/2819945/mod_page/content/1/Bath-AI13-W1-L3-Tic-Tac-Toe.png)  
**An example state in the game of tic-tac-toe.**

If we had a state representation that summarised which cells on the game board were empty, which had been chosen by X, and which had been chosen by O, our agent doesn't need to know the history of all of the moves made so far in order to evaluate what move it should make next. It can gather all the relevant information from the current state of the game board. The Markov property holds.

If, instead, we had a state representation that consisted only of where the most recent two symbols were placed, then the Markov property would not hold. In this case, the state would not summarise all of the past information that our agent needs to evaluate which action to choose next, and knowing the full history would provide additional relevant information.

### Markov Decision Processes (MDPs)

If a sequential decision problem has the Markov property, then it is a **Markov Decision Process (MDP)**. To define an MDP, we need the following:

- A set of environmental states, 𝑆.
- A set of actions available in each state, 𝐴(𝑠),𝑠∈𝑆.
- A transition function defining one-step dynamics, 𝑝(𝑠′∣𝑠,𝑎),𝑠∈𝑆,𝑠′∈𝑆,𝑎∈𝐴(𝑠).
- A reward function defining reward expectations, 𝑟(𝑠,𝑎,𝑠′),𝑠∈𝑆,𝑠′∈𝑆,𝑎∈𝐴(𝑠).
- An initial-state distribution, ℎ(𝑠),𝑠∈𝑆.

If the state-set and action-set are both finite, then we say that the MDP is a **finite MDP**.

The MDP framework enables us to model the type of agent-environment interaction discussed previously in a manner that allows us to think about it mathematically.

Our agent starts off in some initial state, determined by the initial state distribution. Each time-step, our agent finds itself in a state 𝑠∈𝑆, and chooses an action 𝑎∈𝐴(𝑠) based on its current policy. When an action 𝑎 is executed, the environment transitions to some new state 𝑠′∈𝑆 according to the transition model 𝑝(𝑠′|𝑠,𝑎), which gives the probability of transitioning to state 𝑠′ given that our agent has just executed action 𝑎 in state 𝑠. Note that this definition of the transition function allows for both stochastic and deterministic environments to be modelled as MDPs. Finally, our agent earns some reward according to the reward function 𝑟(𝑠,𝑎,𝑠′).

Sometimes, when working with episodic tasks, it can be useful to distinguish between the full set of states and just the non-terminal states. Whenever we need to do this, we will denote the set of non-terminal states as 𝑆+⊂𝑆. At times, it can also be useful for us to write the transition function and reward function jointly as a single function, 𝑝(𝑠′,𝑟∣𝑠,𝑎).

In the literature and in the textbook, you might see slightly different formulations for finite MDPs, or slightly different notation from the one shown above. These differences are usually very minor, and what we cover in this unit should allow you to quite easily understand these alternative formulations and notations when you encounter them.

# Example: Recycling Robot

Completion requirements

We will now walk through how to define a simple sequential decision problem as an MDP.

The problem we're going to consider is the Recycling Robot problem from the course textbook [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496508&forceview=1#footnote-1). This problem is ideal as a first example, because its extreme simplicity means that defining it fully and explicitly as an MDP is not too cumbersome.

In this problem, a mobile robot is tasked with collecting rubbish in an office environment. The robot has sensors for detecting rubbish, an arm for collecting it, a control system, wheels for navigation, and a rechargeable battery. We want our agent to make high-level decisions about how to search for rubbish based on the robot's battery charge level.

There are two states in this environment: **_high_** when the agent's battery has a high charge, and **_low_** when the agent's battery has a low charge. When in the **_high_** state, our agent can choose to **_search_** its environment for rubbish, or to **_wait_** in place for an office worker to manually give it some rubbish. When in the _**low**_ state, our agent can _**search**_ and **_wait_**, but also return to a charging point and **_recharge_** its battery. Our robot is always fully charged from a charging point when it starts.

Based on this, we can define the state set, action set, and initial state distribution for this environment:

- 𝑆={high,low}
- 𝐴(high)={search,wait}
- 𝐴(low)={search,wait,recharge}
- ℎ(high)=1 and ℎ(low)=0

As previously mentioned, the agent can actively move around to find rubbish, or wait in place for an office worker to give it some rubbish. Actively searching for rubbish is the best way to find it, but doing so uses battery charge, whereas waiting in place does not use any battery.

If the agent searches for rubbish, there is a chance that its battery level will fall. If the agent's battery currently has a _**high**_ charge, then searching might cause it to fall to a low charge with probability 0.2, or leave it with a high charge with probability 0.8. Either way, the agent does not run out of battery completely, so the search is counted as successful. If the agent's battery currently has a _**low**_ charge, then searching might cause it to run out of charge completely with probability 0.4, in which case the robot will have to be rescued and recharged by an office worker. Otherwise, it is left with a _**low**_ charge with probability 0.6.

In this environment, our agent receives +1 reward per piece of rubbish that it collects, and it receives −3reward if it ever completely runs out of battery. For the sake of simplicity, let us assume that the agent finds five pieces of rubbish per successful search action, and is given one piece of rubbish per wait action. Recharging when on _**low**_ battery does not give any reward (0), but allows the agent to move from having _**low**_ battery charge to _**high**_ battery charge.

With this description, we have fully defined the Recycling Robot problem as an MDP, specifying the state set, action set, transition dynamics, and rewards. The figure and table below summarise this information.

# Lesson 1 Summary

Completion requirements

In this lesson, we have looked at and introduced notation for describing two types of sequential decision problems, and formalised such problems as MDPs. We have explored

- how sequential decision problems can be either **episodic** or **continuing**,
- introduced notation with which we can describe both episodic and continuing sequential problems,
- discussed what information needs to be summarised in our state representation, in order to ensure that the **Markov property** holds, and
- formulated sequential decision problems formally as **Markov decision processes (MDPs)**.
# Lesson 2 Introduction

Completion requirements

In this lesson, we're going to introduce one of the most fundamental concepts in reinforcement learning: value functions.

Value functions lie at the heart of most reinforcement learning algorithms (including Q-learning, which we've already taken a brief look at). Value functions tell our agent how good it is to be in a given state, or how good it is to take a given action in a given state. This enables our agent to compare states, actions, and even policies.

After this lesson, we will have all the tools we need to start examining reinforcement learning algorithms in more detail.

# Value Functions

Completion requirements

Value functions are incredibly important in reinforcement learning, and lie at the heart of nearly all of the reinforcement learning algorithms we'll cover in this unit. The two types of value functions that we will be looking at are **state-value functions** and **action-value functions**. The concept of **value** is closely tied to the concept of the return, which we have already discussed previously; if the return tells us about a particular sequence of rewards, the value tells us about the **expected return** while following a particular policy.


  
The value of a state 𝑠 under policy 𝜋, denoted 𝑣𝜋(𝑠), is given as as:

𝑣𝜋(𝑠)≐𝐸𝜋{𝐺𝑡∣𝑆𝑡=𝑠}=𝐸𝜋{∑∞𝑘=0𝛾𝑘𝑅𝑡+𝑘+1∣𝑆𝑡=𝑠}

where 𝐸𝜋{} denotes the expected value of a random variable while following policy 𝜋.

Upon first inspection, this might seem quite complicated, but the interpretation is actually quite simple, and we have seen most of this maths before. The value of a state 𝑠 under a policy 𝜋 is simply the **expected return** that our agent will earn if it starts in state 𝑠, and chooses actions according to 𝜋 from that point onward.

An important thing to notice is that, on the left-hand side of this equation, the state value function is explicitly defined with respect to a particular policy 𝜋. This makes sense because acting in one way could lead us to a different return than if we acted differently.

On the right-hand side, we can see that the value of a state is being defined as the expectation of our definition of the total discounted return, which we have discussed previously, if our agent starts in state 𝑠 and follows 𝜋 thereafter.

As a side note, we define the value of any terminal state as 0. This makes sense because the value of a state is the future return that we can expect to earn in the future after visiting it under a given policy; there is no future after visiting a terminal state!

In short, the state-value function answers the question: _"What can I expect my total discounted return to be if I start in state 𝑠 and then act according to a policy 𝜋 thereafter?"_


In a very similar way, we can define the value of taking an action 𝑎 in a state 𝑠 under policy 𝜋, denoted 𝑞𝜋(𝑠,𝑎), as:

𝑞𝜋(𝑠,𝑎)≐𝐸𝜋{𝐺𝑡∣𝑆𝑡=𝑠,𝐴𝑡=𝑎}=𝐸𝜋{∑∞𝑘=0𝛾𝑘𝑅𝑡+𝑘+1∣𝑆𝑡=𝑠,𝐴𝑡=𝑎}

The intuition here is exactly the same as for the state-value function. The action value captures the return our agent can expect to earn if it starts in state 𝑠, takes action 𝑎, and then follows policy 𝜋thereafter.

All that's changed from what we saw in the state value function above is that we're now specifying the action to be taken at the current time-step, 𝑎, instead of leaving it up to the policy 𝜋 (which continues to take control during future time-steps).

Just as is the case with the value of terminal states, we define the value of taking an action in a terminal state as 0.

The action-value function answers the question: _"What can I expect my total discounted return to be if I start in state 𝑠, take action 𝑎, and then act according to a policy 𝜋 thereafter?"_


# Comparing Policies and Optimality

Completion requirements

Now that we know how value functions are defined and have a good feel for what they mean intuitively, we can start to think about what they might be useful for.

#### Comparing policies

---

Firstly, value functions give us a way of comparing different policies, giving us the tools we need to say whether one policy is better or worse than another. Intuitively, if the value of a given state 𝑠 under one policy 𝜋 is higher than that of another policy 𝜋′, then we can expect starting in state 𝑠 and choosing actions according to 𝜋 will lead us to a higher return than choosing actions according to 𝜋′.

We can say that policy 𝜋 is as good as or better than policy 𝜋′ if the value of every state under 𝜋 is greater than the value of every state under 𝜋′. Formally:

𝜋≥𝜋′ if and only if 𝑣𝜋(𝑠)≥𝑣𝜋′(𝑠),∀𝑠∈𝑆

#### Optimal value functions

---

Being able to compare policies allows us to establish an ordering on the set of possible policies. Among the set of all possible policies for solving a given MDP, there will always be at least one that is better than or equal to all of the others. We call such policies **optimal policies**, and denote them 𝜋∗. Intuitively, following an optimal policy starting in any state will lead to a higher expected return than if we followed any non-optimal policy.

By definition, these optimal policies will share a common state-value function, which we call the **optimal state-value function**, denoted 𝑣∗:

𝑣∗(𝑠)=max𝜋𝑣𝜋(𝑠)∀𝑠∈𝑆

Of course, they also share a common action-value function, called the **optimal action-value function**, denoted 𝑞∗:

𝑞∗(𝑠,𝑎)=max𝜋𝑞𝜋(𝑠,𝑎)∀𝑠∈𝑆,∀𝑎∈𝐴(𝑠)


# Deriving Policies from Value Functions

Completion requirements

Perhaps the most useful thing about value functions is that we can use them to derive policies. To illustrate this, we will use an example problem from the course textbook [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496516&forceview=1#footnote-1).

In this problem, our agent finds itself in a gridworld. Each grid square is a unique state, and the agent's goal is to maximise the discounted return (assume 𝛾=0.9). In each state, four actions are available: north, south, east, and west. These actions are reliable, and cause our agent to move in its intended direction with probability 1.0. Actions which would cause the agent to move into a wall leave its state unchanged, but result in a reward of −1. Actions which move the agent out of the state marked '𝐴' cause the agent to move to the state marked '𝐴′', and result in a reward of +10. Actions which move the agent out of the state marked '𝐵' cause the agent to move to the state marked '𝐵′', and result in a reward of +5. Rewards for all other actions are 0. The environment is shown in the left panel of the figure below.


**A Gridworld problem, with its associated optimal state-value function and optimal policy.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496516&forceview=1#footnote-2)

Given the optimal state-value function for this problem, shown in the centre panel of the above figure, how might we derive the optimal policy in each state? Let us consider the state below state 𝐵′, marked with a question mark in the left figure.

What is the value of the states we can reach?

- If we choose to go north, we will receive an immediate reward of 0 and reach a state with a value of 16.0.
- If we choose to go west, we will receive an immediate reward of 0 and reach a state with a value of 16.0.
- If we choose to go east, we will receive an immediate reward of 0 and reach a state with a value of 13.0.
- If we choose to go south, we will receive an immediate reward of 0 and reach a state with a value of 13.0.

The states that we can reach by going north or west from this state have the highest value, according to the optimal state-value function. This is what we want, because travelling to these states will give us the best expected discounted return. So, in this state, the optimal policy would be to move either north or west.

What we have done here is act greedily with respect to the optimal state-value function. We've done a one-step look-ahead to all of the states that we can reach from our current state, and chosen action(s) which lead to the highest immediate reward and the next-state(s) with the highest value(s). Applying this greedy one-step look-ahead rule to every state, we can derive the optimal policy for this problem, shown in the right panel of the above figure (optimal actions are represented by directional arrows).

In fact, given the optimal state-value function for any MDP, and the ability to look ahead one step to see what next-states and rewards our actions will lead to, we can derive the optimal policy. If we have access to the optimal action-value function, we do not even need to do a one-step look-ahead, because we would know the value of each of the actions available in the current state, and could just pick the highest-valued action directly.

This is why value functions are so important in reinforcement learning: if we know the optimal value function(s) for a given MDP, we can use them to derive an optimal policy. Because of this, many reinforcement learning algorithms aim to learn value functions, and then use those value functions to derive their policies. We call such methods value-based reinforcement learning methods, and they will make up the vast majority of the methods we'll look at during this unit.

# Lesson 2 Summary

Completion requirements

In this lesson, we introduced the concept of **value functions** and formally defined **state-value** and **action-value** functions. Then, we examined how these value functions can be used to compare policies. We introduced the concept of an **optimal policy**, and discussed the how we can derive such policies from their corresponding value functions. We discussed

- that **state-value functions** tell us the return we can expect to earn is we start in a given state and then follow a given policy thereafter,
- that **action-value functions** tell us the return we can expect to earn is we start in a given state, take a given action, and then follow a given policy thereafter,
- how we can use value functions to **compare policies** to each other,
- how, for any MDP, there is a set of **optimal policies** which are better than all others,
- how all of the optimal policies for a given MDP share common **optimal state-value functions**and **optimal action-value functions**, and
- how we can use optimal value functions to **derive optimal policies**.
# Lesson 3 Introduction

Completion requirements

In this lesson, we will be looking at the Bellman equations. The Bellman equations lie at the heart of the Dynamic Programming methods that we'll be looking at later this week, and tell us how the value of one state can be defined in terms of the values of its possible successors.

You may have encountered the Bellman equations in previous units, likely while looking at dynamic programming methods in isolation. This lesson should act as a solid review of the ideas you will have been exposed to previously, as well as a chance to revisit these ideas with additional context through the lens of reinforcement learning.

# The Bellman Equations

Completion requirements

As we have previously discussed, it is possible to derive a recursive definition for the return: the return at time 𝑡 can be decomposed into the sum of the immediate reward and the return at time 𝑡+1. The **Bellman equations** use this recursive definition to define a useful relationship between the value of a state and the values of its possible successor states.

Let us derive the **Bellman equation** for 𝑣𝜋 from first principles, starting with the definition of 𝑣𝜋 that we have seen previously:

𝑣𝜋(𝑠)≐𝐸𝜋{𝐺𝑡∣𝑆𝑡=𝑠}

We can decompose the return at time 𝑡, 𝐺𝑡, into the immediate reward and the discounted return at time-step 𝑡+1:

𝑣𝜋(𝑠)=𝐸𝜋{𝑅𝑡+1+𝛾𝐺𝑡+1∣𝑆𝑡=𝑠}

Let's now expand the expectation. Recall that the expected value of a random variable is the weighted sum of all possible values of that random variable, where the weights are the probabilities of each outcome occurring. In our case, the probability of a given outcome occurring depends on the actions our agent chooses (its policy 𝜋(𝑎|𝑠)) and the environment's dynamics (the transition function 𝑝(𝑠′,𝑟′|𝑠,𝑎)).

𝑣𝜋(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝐸𝜋{𝐺𝑡+1∣𝑆𝑡+1=𝑠′}]=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣𝜋(𝑠′)]

What we have ended up with is the Bellman equation for 𝑣𝜋. It might look quite complex at first, but the intuition behind it is actually quite simple. It's the weighted sum of the value of all possible outcomes (decomposed into the immediate reward and the value of the next-state) weighted by the probability of each outcome occurring.

It can be helpful to try and interpret this visually, and this is often done through the use of **backup diagrams** (so-called because they illustrate the backup, or update, rules used in many reinforcement learning methods).

The backup diagram is shown below, and illustrates that the value of a state 𝑠 is based on the weighted sum of the immediate rewards 𝑟 and the value of the next-states 𝑠′ that can be reached from it, via actions 𝑎.

Remember, all the Bellman equation for 𝑣𝜋 tells us is how to define the value of one state in terms of its possible successors.

# The Bellman Optimality Equations

Completion requirements

The Bellman equation for 𝑣𝜋 states that the value of a state 𝑠 is equal to the expected discounted value of its next-states, plus whatever immediate reward is earned along the way. This constraint is true for all of the states in a given MDP, so what we actually find ourselves with is a set of linear equations (one for each state 𝑠∈𝑆). Each equation tells us the expected value of a different state, in terms of its possible successors and immediate rewards. We can solve this set of linear equations to find 𝑣𝜋.

As with any other value function, the optimal value function 𝑣∗ must also satisfy this constraint. However, because 𝑣∗ is the value function for the optimal policy 𝜋∗, we can write the Bellman equation for 𝑣∗ in a slightly different way. We call this the **Bellman optimality equation** for 𝑣∗:

𝑣∗(𝑠)=max𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣∗(𝑠′)]

Superficially, this looks very similar to the Bellman equation for 𝑣𝜋 that we derived previously. However, because we are now working with the optimal policy, we only consider the action that results in the highest expected return, because this is the action that the optimal policy would choose. Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return of the best action available in that state.

The backup diagram associated with the Bellman optimality equation for 𝑣∗ is shown below.

**Notice** the arc over the action choices: this denotes that we are taking a maximum, instead of taking a weighted sum.

**The backup diagram associated with the Bellman optimality equation for 𝑣∗.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496523&forceview=1#footnote-1)

Just as before, if we have an environment with 𝑛 states, then we have a set of 𝑛 of these equations with 𝑛unknowns, and the unique solution for this set of equations is 𝑣∗. If we have perfect knowledge of the environment's dynamics, 𝑝(𝑠′,𝑟|𝑠,𝑎), then we can solve this set of equations for 𝑣∗.

Once we have solved for 𝑣∗, it is easy to derive an optimal policy using the one-step look-ahead method that we have discussed previously.

Unfortunately, because the Bellman optimality equation includes a max operator, we're no longer working with a system of linear equations, which makes finding a solution more complicated; in most cases, we will be limited to approximating a solution using iterative methods instead of finding nice, closed-form solutions.

Dynamic Programming methods, which we'll be looking at soon, give us one way of solving the Bellman equations. Most reinforcement learning methods can be understood as finding approximate solutions to the Bellman optimality equation.

The amount of maths in this lesson might have been a little difficult to digest. So, let's finish off by looking at a concrete example. Once again, we'll be using a variation of the Gold Gridworld environment, and will take a look at what the Bellman optimality equation for 𝑣∗ looks like for a single state.

We have nine states (numbered 1 to 9), one corresponding to each grid square, and the four actions 'North', 'South', 'East', and 'West' are available in each state. Our agent starts in state 2, and state 9is a terminal state. Actions that would move the agent into a wall result in no movement, but time advances as normal.

To keep the maths simpler for this example, we will assume that all actions are reliable: actions succeed as intended with probability 1.0. If our agent transitions to state 9, it earns a reward of +10. Otherwise, it earns a reward of −1.

For this example, we're going to be looking at state 8. Let's first look at the possible next-states and rewards that can be reached from this state.

- Going **_North_** bounces us back to state 8 and we recieve a reward of −1.
- Going **_West_** takes us to state 7 and we recieve a reward of −1.
- Going **_East_** takes us to state 9 and we recieve a reward of +10.
- Going **_South_** takes us to state 5 and we recieve a reward of −1.

With this information, we can write out the Bellman optimality equation:

𝑣∗(8)=max{1[−1+𝛾𝑣∗(8)],“North” Action1[−1+𝛾𝑣∗(7)],“West” Action1[+10+𝛾𝑣∗(9)],“East” Action1[−1+𝛾𝑣∗(5)]}“South” Action

If we wrote out similar equations for the other eight states and solved for 𝑣∗ (using dynamic programming methods, for example), we'd end up with a concrete value for each state, and could easily derive the optimal policy.

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Lesson 3 Summary

Completion requirements

This lesson, we have taken a look at the Bellman equations. These equations define relationships between the values of states and their successors. As we shall see soon, they also form the basis of Dynamic Programming methods. We have discussed

- how the **Bellman equation for** 𝑣𝜋 relates the value of a state to the values of its possible successor states,
- how, for a given MDP, we can define a set of these linear equations, one for each 𝑠∈𝑆, which can be solved for 𝑣𝜋,
- how the Bellman equation for 𝑣∗ can be written in a special form, known as that **Bellman optimality equation** for 𝑣∗, and
- how many reinforcement learning methods can be viewed as finding approximate solutions to the Bellman optimality equation.

# Lesson 4 Introduction

Completion requirements

In this lesson, we'll be taking a detailed look at our first concrete class of solution methods: **D****ynamic Programming** (**DP**) methods.

Dynamic programming methods assume that a perfect model of the environment is known, and use it to iteratively compute optimal value functions and derive optimal policies using the Bellman equations.

Unfortunately, because they require the dynamics of the target environment to be fully known, dynamic programming methods are of limited use when dealing with more complex problems. However, they are still of great theoretical importance, and serve as the foundation for the other families of methods we'll be looking at over the next few weeks.


![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Iterative Policy Evaluation

Completion requirements

Over the course of this unit, we are going to be introducing various families of methods for solving sequential decision problems modelled as MDPs. When first introducing a class of solution methods, we will start off by discussing how those methods treat the problem of computing the value function 𝑣𝜋 for a given policy 𝑣𝜋. We call this the **policy evaluation** or **prediction** problem. We're not trying to find an optimal policy yet, we're just trying to find the value function for a given policy.

This is exactly how we are going to start our discussion of **D****ynamic Programming (DP)** methods. If our agent starts in state 𝑠 and acts according to policy 𝜋, what return can it expect to earn? How do DP methods compute 𝑣𝜋?

Recall the Bellman equation for 𝑣𝜋 that we introduced previously:

𝑣𝜋(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣𝜋(𝑠′)]

For an environment with 𝑛 states, we have a set of 𝑛 of these equations which we can solve for 𝑣𝜋. DP is a class of iterative methods for solving the Bellman equations, assuming that we have perfect knowledge of the environment's dynamics 𝑝(𝑠′,𝑟|𝑠,𝑎).

We can very easily define an iterative update rule based on the Bellman equation for 𝑣𝜋:

𝑣𝑘+1(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣𝑘(𝑠′)]

where 𝑣𝑘 is our approximation of 𝑣𝜋 after 𝑘 iterations.

Starting with some initial estimate 𝑣0, such as assuming that the value of all states is zero under policy 𝜋, we can apply the above update rule again and again in every state 𝑠∈𝑆. After each iteration, we'll have a new estimate based on our previous estimate. Eventually, our estimate will converge (i.e., the difference between 𝑣𝑘 and 𝑣𝑘+1 will fall below some threshold Δ) and we'll be left with 𝑣𝑘+1≈𝑣𝜋.

𝑣0→𝑣1→…→𝑣𝑘→𝑣𝑘+1→…→𝑣𝜋

The algorithm that we have just described is called **I****terative Policy Evaluation**, and its pseudocode is presented below.

Iterative policy evaluation, for estimating 𝑉≈𝑣𝜋

 Input 𝜋, the policy to be evaluated  Algorithm parameter: a small threshold 𝜃>0 determining accuracy of estimation  Initialize 𝑉(𝑠) arbitrarily, for 𝑠∈𝑆, and 𝑉( terminal ) to 0 Loop: Δ←0 Loop for each 𝑠∈:𝑣←𝑉(𝑠)𝑉(𝑠)←∑𝑎𝜋(𝑎∣𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝑎)[𝑟+𝛾𝑉(𝑠′)]Δ←max(Δ,|𝑣−𝑉(𝑠)|) until Δ<𝜃

**The pseudocode for the iterative policy evaluation algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-1)

As you can see, given some input policy 𝜋 and an initial guess for its corresponding value function, this algorithm iteratively applies the update rule that we defined above until the estimate of 𝑣𝜋 converges. An example of this algorithm being applied to a simple gridworld can be found in this week's required reading material.

Let us briefly highlight two key features of updates made by dynamic programming methods, such as iterative policy evaluation, so that we can compare and contrast them with other solution methods that we'll encounter later.

𝑣𝑘+1(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣𝑘(𝑠′)]

First, notice that our estimate for 𝑣𝑘+1(𝑠) relies on our existing estimate of 𝑣𝑘(𝑠′).

We refer to this situation, where we base a new estimate on an existing estimate, as **bootstrapping**.

Secondly, because dynamic programming methods assume perfect knowledge of the environment's dynamics, the updates they perform are able to take into account every possible next-state 𝑠′ and reward 𝑟 that we can possibly transition to from state 𝑠, along with all of the associated probability distributions.

We refer to this kind of update which takes into account perfect information as a **full backup**.

Later in the course, we will explore classes of solution methods that employ different types of backups. Some will use bootstrapping, while others will not.


Let's try applying this algorithm to an example environment. We'll use a simple 4x4 gridworld [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-2)environment for this example, illustrated below.

The top-left and bottom-right shaded grid squares are terminal states. In each state, our agent has access to four reliable actions: 'North', 'South', 'East', and 'West'. As usual, actions which would move the agent into a wall leave its state unchanged, but time advances. The reward is −1 for all actions, and this is an undiscounted (i.e., 𝛾=1) episodic problem.

This is the environment in which we will be applying the **iterative policy evaluation** algorithm. The policy that we are going to be evaluating is the equiprobable random policy (i.e., it assigns probability 0.25 to each of the four available actions in each state). For our initial estimate for 𝑣0, let us guess that the value of every state is 0.0, as shown below.

**4x4 gridworld with numbered states.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-4)

Let's apply one iteration of iterative policy evaluation, using the update rule that we derived earlier:

𝑣1(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣0(𝑠′)]

If we plug in the next-states, rewards, and action probabilities, we find that this update is the same for all states in this first iteration:

𝑣1(𝑠)=0.25[−1+0]+0.25[−1+0]+0.25[−1+0]+0.25[−1+0]=−0.25+−0.25+−0.25+−0.25=−1∀𝑠∈𝑆+(1)

Notice that we only iterate over the set of non-terminal states. Remember that, by definition, the value of a terminal state must be zero.

So, after this first iteration, we end up with the following estimate, 𝑣1.

**Our estimate of the value of each state in the 4x4 gridworld under the random policy after applying one iteration of policy evaluation.** [**[2]**](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-5)

Repeating this same process again two more times - plugging the numbers into our update equation from our existing estimates, next-states, rewards, and action probabilities - we get the two estimates, 𝑣2 and 𝑣3, shown below. Try plugging the numbers from 𝑣1 into our update rule for a few different states, and see whether you get the correct answer for 𝑣2.

**Our estimate of the value of each state in the 4x4 gridworld under the random policy after applying two and three iterations of policy evaluation.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-6)

If we were to continue applying our iterative update rule over and over again, we would eventually reach a point where our successive updates would change very little. At this point, we'd say that our estimate had converged, and we'd be left with a good approximation for the true value function 𝑣𝜋. In the limit, if we were to run an infinite number of iterations, we'd eventually find the true value function.

**Our estimate of the value of each state in the 4x4 gridworld under the random policy after applying infinitely many iterations of policy evaluation. At this point, 𝑣∞=𝑣𝜋.** [**[2]**](https://moodle.bath.ac.uk/mod/page/view.php?id=1496529&forceview=1#footnote-7)

# Policy Improvement

Completion requirements

So far, we have looked at iterative policy evaluation, a dynamic method for solving the prediction problem: estimating the value function 𝑣𝜋 for a given policy 𝜋.

Having solved that simpler problem, we're now ready to look at how dynamic programming methods address the harder problem of finding the optimal policy 𝜋∗ for a given MDP. This is known as the **control problem**.

Let's say that our agent is currently acting according to some policy 𝜋, and we have used iterative policy evaluation to find its corresponding state-value function 𝑣𝜋. How can our agent improve its policy? This process is known as **policy improvement**.

Assuming that we have perfect knowledge of the environment's dynamics, there is a straightforward way to improve our agent's policy based on its state value function. For each state 𝑠∈𝑆, we can calculate the value of taking each of the actions 𝑎∈𝐴(𝑠):

𝑞𝜋(𝑠,𝑎)=∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝑎)[𝑟+𝛾𝑣𝜋(𝑠′)]

Remember, this quantity tells us the expected return after taking action 𝑎 in state 𝑠 and following policy 𝜋thereafter. If, for some action 𝑎, we find that 𝑞𝜋(𝑠,𝑎)>𝑣𝜋(𝑠), then taking 𝑎 in state 𝑠 is better than taking whatever action 𝜋 would take. A new policy 𝜋′ that chooses 𝑎 in state 𝑠 will therefore be an improvement on 𝜋 (i.e., 𝜋′>𝜋).

In other words, acting greedily with respect to 𝑞𝜋 — that is, choosing the action(s) in each state that maximise the expected reward plus next-state value — produces a policy that is guaranteed to be at least as good as the current one. This result is known as the **policy improvement theorem**.

You may have already noticed, but this is very similar to the one-step look-ahead process that we have used previously to derive optimal policies from optimal value functions.

# Policy Iteration

Completion requirements

So far, we have answered two questions:

- **Policy evaluation**/**p****rediction**: Given a policy 𝜋, how can I compute its value function 𝑣𝜋?
- **Policy improvement**: Given the value function 𝑣𝜋 for a policy 𝜋, how can I derive a better policy?

What we're going to do now is combine these two steps to produce a dynamic programming algorithm that starts with an arbitrary policy and, after many iterations, eventually converges upon an optimal policy. Let's say that we started with an arbitrary policy 𝜋0, such as the equiprobable random policy. In order to generate a new policy 𝜋1 which is better than 𝜋0 using dynamic programming methods, we could do the following:

1. Use **I****terative Policy Evaluation** to find the value function 𝑣𝜋0.
2. Use **p****olicy improvement** to generate a policy 𝜋1 that is an improvement over 𝜋0.

We now have a new policy, 𝜋1. We can apply this same two-step process of applying iterative policy evaluation and policy improvement in order to find a policy 𝜋2 that improves on 𝜋1. In fact, if we repeat this two-step process enough times, we are guaranteed to (eventually) end up with the optimal policy 𝜋∗. Along the way, we'd go through a sequence of policies and corresponding value functions:

𝜋0→𝑣𝜋0→𝜋1→𝑣𝜋1→𝜋2→𝑣𝜋2→…→𝜋∗→𝑣𝜋∗

This repeated two-step process of applying iterative policy evaluation and policy improvement is known as the **P****olicy Iteration** algorithm, the pseudocode for which is shown below. Given some arbitrary initial policy and value function estimates, this algorithm will eventually converge to the optimal policy 𝜋∗ with probability 1.0.

Policy Iteration (using Iterative Policy Evaluation) for estimating 𝜋≈𝜋∗

 1. 2. 3. Initialization 𝑉(𝑠)∈ℝ and 𝜋(𝑠)∈(𝑠) arbitrarily for all 𝑠∈𝑆;𝑉( terminal )≐0 Policy Evaluation  Loop: Δ←0 Loop for each 𝑠∈𝑆:𝑣←𝑉(𝑠)𝑉(𝑠)←∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝜋(𝑠))[𝑟+𝛾𝑉(𝑠′)]Δ←max(Δ,|𝑣−𝑉(𝑠)|) until Δ<𝜃 (a small positive number determining the accuracy of estimation)  Policy Improvement  policy-stable ← true  For each 𝑠∈𝑆: old-action ←𝜋(𝑠)𝜋(𝑠)←argmax𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝑎)[𝑟+𝛾𝑉(𝑠′)] If old-action ≠𝜋(𝑠), then policy-stable ← false  If policy-stable, then stop and return 𝑉≈𝑣∗ and 𝜋≈𝜋∗; else go to 2

**The pseudocode for the policy iteration algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496531&forceview=1#footnote-1)

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Value Iteration

Completion requirements

The **policy iteration** algorithm is a fantastic algorithm which, given some arbitrary initial policy 𝜋0, allows us to find iteratively better policies until we are left with the optimal policy 𝜋∗. We will now look at an alternative: **V****alue Iteration**.

Value iteration is another dynamic programming algorithm that uses many of the same ideas that led us to policy iteration, but which is much simpler.

The dynamic programming methods we've looked at so far have based their update rules on the Bellman equation for 𝑣𝜋. In contrast, value iteration uses the Bellman optimality equation for 𝑣∗ as the basis for its updates:

𝑣𝑘+1(𝑠)=max𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣𝑘(𝑠′)]

Previously, we saw policy iteration estimate the value function of the current policy, and then derive an improved policy based on that estimated value function. Each iteration consisted of two separate steps: first, policy evaluation was run until convergence, and then policy improvement was used to derive an improved policy.

Now, we can see value iteration incorporating the greedy action-selection into its value updates through the use of the max operator from the Bellman optimality equation. By using the Bellman optimality equation as the basis for its update rule, value iteration is able to merge policy evaluation and policy improvement into a single step, where possible improvements to the agent's policy are effectively taken into account every single time that the value function is updated.

Another way of understanding how value iteration works is by thinking of each iteration of value iteration as performing one step of policy evaluation followed by one step of policy improvement, instead of running policy evaluation and policy improvement until convergence each time.

The pseudocode for the value iteration algorithm is shown below. Notice that it only requires us to provide an arbitrary initial estimate for the value function 𝑣0, and does not derive a policy until the very end, when the value function estimate has converged upon the optimal value function 𝑣∗.

Value Iteration, for estimating  𝜋≈𝜋∗

 Algorithm parameter: a small threshold 𝜃>0 determining accuracy of estimation  Initialize 𝑉(𝑠), for all 𝑠∈+,arbitrarily except that 𝑉( terminal )=0 Loop:  |Δ←0 | Loop for each 𝑠∈: |𝑣←𝑉(𝑠) |𝑉(𝑠)←max𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝑎)[𝑟+𝛾𝑉(𝑠′)] |Δ←max(Δ,|𝑣−𝑉(𝑠)|) until Δ<𝜃 Output a deterministic policy, 𝜋≈𝜋∗, such that 𝜋(𝑠)=argmax𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟∣𝑠,𝑎)[𝑟+𝛾𝑉(𝑠′)]

**The pseudocode for the value iteration algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496532&forceview=1#footnote-1)

#### Efficiency of DP Methods

Despite today's computers allowing dynamic programming methods like value iteration to be applied to problems with millions of states, it is not usually feasible to apply them to many of the real-world problems that we are interested in solving. In the worst case, the time complexity of a dynamic programming method trying to find an optimal policy is polynomial in the number of states and actions in the target problem. Given that a problem like chess has somewhere in the region of 1045 states, this limits their applicability greatly.

This is often referred to as the 'Curse of Dimensionality', a term coined by Bellman when he was first working on DP methods. You may well have encountered its effects in other areas of machine learning or statistics when trying to tackle problems with high-dimensional data.

Worse still, dynamic programming assumes full knowledge of the environment’s transition dynamics 𝑝(𝑠′,𝑟|𝑠,𝑎). Without this information, we cannot apply their Bellman-style update rules. This poses a major problem because, in most real-world reinforcement learning tasks, we simply don't have access to this information. What we would really like is for agents to learn directly from experience, starting with no prior knowledge of their environment.

# Generalised Policy Iteration

Completion requirements

We will finish this lesson by introducing the concept of **G****eneralised Policy Iteration (GPI)**, a term that is used to refer to any interaction between a policy evaluation process and a policy improvement process, regardless of the granularity of either step.

This idea has been present in both policy iteration and value iteration, but we haven't explicitly discussed it in its own right. 

We start off with an arbitrary policy and derive an estimate for its value function. Then, we use this value function estimate to derive a better policy. If we repeat this process enough times, we are guaranteed to converge to the optimal policy and value function.

**A diagram illustrating the process of generalised policy iteration.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1496533&forceview=1#footnote-1)

The goal of GPI is to solve the **control problem**: finding an optimal policy 𝜋∗.

Clearly, policy iteration is an instance of GPI. It is a DP method that runs iterative policy evaluation until convergence in order to find an estimate of its current policy's value function, and then derives an improved policy by acting greedily with respect to that value function estimate. Likewise, value iteration is also an instance of GPI; the two processes are still present, but value iteration only performs one step of policy evaluation before choosing an action greedily.

Almost all RL algorithms can be seen as instances of GPI. This is why, as we move onto other classes of solution methods, we will often start by looking at how to solve the policy evaluation/prediction problem: it is an essential first step in this GPI process.

Try to identify the GPI cycle in all of the control algorithms we see throughout the rest of this course!

# Lesson 4 Summary

Completion requirements

This lesson, we have introduced our first concrete class of solution methods for sequential decision problems: **d****ynamic programming** methods. These approaches use perfect knowledge of a given environment's dynamics to iteratively solve the Bellman equations and compute value functions and policies. We have discussed

- how we can use **i****terative policy evaluation** to solve the **p****olicy evaluation** problem of computing the state-value function 𝑣𝜋 for a given policy 𝜋,
- how we can use **policy improvement** to derive a policy 𝜋′ which is an improvement over another policy 𝜋 given its state-value function 𝑣𝜋,
- how we can combine these policy evaluation and policy improvement steps to create a single DP algorithm, **p****olicy iteration**, for finding the optimal policy 𝜋∗,
- how **v****alue iteration** simplifies policy iteration by combining policy evaluation and policy improvement into a single step,
- that any interaction between a policy evaluation process and a policy improvement process is known as **g****eneralised policy iteration (GPI)**,
- that the goal of generalised policy iteration is to find an optimal policy, a problem known as the **control problem**, and
- how most reinforcement learning algorithms can be described as instances of generalised policy iteration.

# Weekly Summary

Completion requirements

This week, we introduced several fundamental concepts in reinforcement learning and took an in-depth look at our first concrete family of methods for solving sequential decision problems.

We started by formally defining our agent's interaction with its environment as a **Markov Decision Process** (**MDP**). To define a sequential decision problem as an MDP, we need to define five things: a set of states, a set of actions, a transition function, a reward function, and an initial state distribution.

We then introduced **value functions**, which are useful functions that tell our agent the return it can expect to earn after starting in a given state or after taking a given action in a given state, while following a given policy. We discussed how these value functions can be used to compare policies and established the concept of **optimal policies**, which achieve a return greater than or equal to that of all other policies in every state.

Leading on from value functions, we introduced the **Bellman equations**. These equations define important recursive relationships between the values of states/state-action pairs and their successors. Solving the Bellman equation for 𝑣𝜋 gives us the state-value function for a given policy 𝜋, which we can use to perform policy improvement. Solving the Bellman optimality equation for 𝑣∗ gives us the optimal state-value function, from which we can derive an optimal policy. These Bellman equations form the basis of the updated rules seen in many reinforcement learning methods.

Finally, we looked at our first concrete class of solution methods: **dynamic programming** methods. These methods use iterative update rules derived from the Bellman equations to solve sequential decision problems. We discussed how dynamic programming methods rely on having access to a perfect model of our agent's environment, and use bootstrapping (i.e., they base their value estimates on previous value estimates). We examined the **iterative policy evaluation algorithm**, which solves the policy evaluation/prediction problem, and the **policy iteration** and **value iteration** algorithms, which solve the control problem of finding an optimal policy 𝜋∗ for a given MDP.

One key limitation of dynamic programming methods is that they assume access to a perfect model of our agent's environment. Unfortunately, this assumption is often unrealistic. Next week, we will introduce a second family of solution methods that enable our agent to learn online using only its own experience, without assuming any prior knowledge about its environment.