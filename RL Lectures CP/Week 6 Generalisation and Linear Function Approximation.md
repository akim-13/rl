
# Lesson 1 Introduction

Completion requirements

In this lesson, we will be discussing a particular property of the reinforcement learning algorithms we have studied so far in this unit. Specifically, we will consider how they are all **tabular**, meaning that they learn and store value estimates for each state/state-action pair separately from each other in a table-like fashion.

We will discuss why, despite working well when applied to simple problems with small, discrete sets of states and actions, these tabular methods do not scale well to more complex problems.

# Tabular Methods

Completion requirements

So far in this unit, we have seen several reinforcement learning algorithms for solving both prediction and control problems.

In terms of control algorithms, we have seen dynamic programming methods such as value iteration, Monte Carlo methods such as on-policy Monte Carlo control, and temporal-difference methods such as Sarsa and Q-Learning. All of these methods are instances of generalised policy iteration (GPI), in which a policy evaluation and policy improvement process interact to learn an optimal policy.

All the algorithms we have looked at so far are **tabular** methods.

This means that we can represent the value function they learn as a table, with one entry for each state or state-action pair.

Consider the 5×5 gridworld shown below, where the agent must navigate from the start state to the goal state as quickly as possible. As usual, the agent has access to actions allowing it to move 'North', 'South', 'East' or 'West' each time step.

**A simple 5×5 gridworld.**

The control algorithms we have looked at so far would learn a single action value for each state-action pair in this environment. We have 25 states and 4 actions per state, so our agent's table would contain 25×4=100 separate entries. Assuming we used a different number between 0 and 24 to identify the states in this environment, our agent's table of action values would look like the table shown below.

|State|Action|𝑄(𝑠,𝑎)|
|---|---|---|
|0|N|𝑄(0,𝑁)|
|0|S|𝑄(0,𝑆)|
|0|E|𝑄(0,𝐸)|
|0|W|𝑄(0,𝑊)|
|…|…|…|
|24|W|𝑄(24,𝑊)|

**A table of action-values with 100 entries for an agent learning in the 5×5 gridworld.**

**Storing value estimates separately in this tabular fashion has significant consequences.**

- Firstly, it means that we have to store information about every state-action pair separately in memory. This means that if our agent's environment has a very large state space, we are likely to need a lot of memory to store its value estimates.
- Secondly, and perhaps more significantly, if our agent learns something about one state-action pair, that learning doesn't immediately affect any other state-action pairs. This means that, to learn about a state-action pair, our agent needs to actually experience it.

For the simple 5×5 gridworld shown above, our agent only needs to store 100 separate action-values, so memory should not be an issue. As you can imagine, it will also be relatively easy for our agent to navigate between different regions of the state-space, so it is quite likely that it will experience all possible state-action pairs through random exploration in a relatively short amount of time.

However, what happens when we stop looking at such small problems and turn our attention to problems with larger state-spaces? Imagine that our agent found itself in a 1000×1000 gridworld instead of a 5×5 one. Now, the agent will need to learn and store 4,000,000 separate action values. Not only will this take up much more memory, but it is also much less likely that our agent will be able to visit or revisit any state-action pair through random exploration.

Ideally, our agent's memory requirements should not increase with the number of state-action pairs it observes, and it should be able to **generalise** what it learns about states it has seen to other, similar states it has not.

# Tabular Methods at Scale

Completion requirements

We will now look at three example problems and discuss whether or not it is feasible to apply tabular reinforcement learning methods to solve them.

The keyword here is feasible. Of course, if we had an infinite amount of memory to store value estimates and an infinite amount of training time, we could apply tabular methods to solve problems with very large discrete state spaces. However, in reality, we have neither of these things!

Consider the game of tic-tac-toe, where two players (X and O) take turns to place their symbol in empty cells on a 3×3 grid. The first player to create a horizontal, vertical or diagonal line of their symbol wins. Winning earns a reward of =1, losing earns a reward of −1, and drawing earns a reward of 0.

An example tic-tac-toe state is shown below.

**An example state in the game of tic-tac-toe.**

Using a state representation that does not account for symmetry in the state-space, tic-tac-toe has around 5500 valid states. Given that the number of actions per state is at most nine, modern computers would obviously have no problem storing a separate action-value estimate for every unique state-action pair in this environment.


### Think Point


##### Take a moment to think about the actions available in this environment. It would be harder for our agent to visit and revisit every state-action pair in this environment than to navigate the simple gridworld we considered earlier, because each action is inherently irreversible. Despite this, because the state space is still quite small, it is still very feasible to solve tic-tac-toe using tabular methods.

Although it is feasible to solve this problem using tabular methods, there are also plenty of opportunities to improve how quickly our agent learns. In this environment, there are many ways that our agent can apply what it knows about one state to learn about other, similar states. For example, consider the three states shown below:

**Three example states in the game of tic-tac-toe. These three states are rotations of each other.**

Each of these three states is simply a rotation of the first state shown above. Ideally, if our agent learned the optimal policy for one of these states, we would like it to generalise that knowledge and apply it to the other three. Using a tabular method, we learn and store information about each state separately, so we wouldn't necessarily be able to do that (again, depending on our state representation).

So, despite it being feasible to solve this problem using tabular methods, it would still likely be useful for our agent to generalise what it knows about one state to other, similar states.

Consider the game of Tetris; an example state is shown below. This game requires the agent to decide how to position a falling block into a grid. If the agent fills an entire row of the grid with blocks, the row is cleared and the agent earns a reward. If the agent runs out of space to place blocks, the game ends.

**An example state in the game of Tetris.** [[3]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512513&forceview=1#footnote-3)

Based on the size of the board (10×20) and the number of legal falling blocks (7), a very crude upper bound for the size of Tetris' state-space is 7×10200 states. To put this number into context, it is probably larger than the number of atoms that make up planet Earth. Clearly, it would not be feasible to store information about every state-action pair in this environment separately. Nor would it be possible to visit and re-visit every one of these states in a reasonable amount of time.

To solve this problem, we cannot use tabular methods. We need to find a way for our agent to avoid storing information about each state-action pair separately, and to generalise what it learns about one state-action pair to other similar state-action pairs (even those it has never visited before).

The final problem that we will discuss this lesson is the Mountain Car problem, an illustrated problem. The goal in this environment is for our agent to drive a car from the bottom of a valley up to a goal on the hill to its right. However, the car is underpowered, and cannot drive all the way up the hill using its engine power alone. Our agent can drive full-throttle forwards, full-throttle backwards, or set the throttle to zero. In order to reach the top of the right hill, the agent must drive away from the goal up the left hill, and use gravity to help it get enough momentum to reach the top of the hill on the right.

**An illustration of the Mountain Car problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512513&forceview=1#footnote-4)

Unlike all the environments we have seen so far, whose states have been discrete, the Mountain Car problem features a continuous state space. States in this environment are represented by two continuous variables: the car's position on the 𝑥-axis (−1.2<𝑥<0.5) and its velocity (−0.07<𝑣<0.07).

The fact that the state variables are continuous is a problem for tabular methods, because an agent is very unlikely to ever revisit the same continuous state.

# Lesson 1 Summary

Completion requirements

In this lesson, we have discussed why the reinforcement learning methods that we have examined so far in this course might not scale well when applied to environments with large state spaces. We also examined a number of example environments where it would not be feasible to apply tabular methods, or where generalising learning across similar states would be desirable. We have discussed

- what it means for a reinforcement learning algorithm to be **tabular**,
- why tabular reinforcement learning methods do not scale well to environments with larger state spaces, and
- whether it is feasible to apply tabular methods to a number of example methods.
# Lesson 2 Introduction

Completion requirements

In the last lesson, we discussed how tabular reinforcement learning methods do not scale well when applied to sequential decision problems with large state spaces. This is because our agent can only reasonably expect to visit a small subset of the entire state space in such cases, and storing information about every state separately would require too much memory.

To tackle problems with larger state spaces, our agent needs to be able to do the following:

- It needs to be able to generalise what it knows about states it has visited to other states.
- It needs to do this without storing information about every state separately.

We will now look at how we can use **function approximation** to achieve both of these goals. You will have encountered function approximation previously when studying supervised learning. At its simplest, function approximation means using training examples to learn an approximation for a given function.

As usual, we will start by considering the prediction problem, where our agent attempts to learn a state-value function 𝑣𝜋(𝑠) which maps states to state-values. So far, using tabular methods, we have represented this state-value function as a table and have learned and stored each value separately.

Using function approximation, we represent this state-value function as a real-valued function with **parameter vector** 𝜽. This function takes a state as its input and outputs its value. Instead of learning individual values for each state, we only store and update our function's parameters 𝜽. Our aim is to update our function's parameters so that our function outputs correct values for as many states as possible.

Assuming we have fewer parameters than states in the state space, changing one parameter will affect the values of many states. This helps us achieve generalisation and stops us from having to worry about the memory requirements of storing information about every state separately. We describe this as approximating the true value function 𝑣𝜋(𝑠) with some function 𝑣̂(𝑠,𝜽) with parameters 𝜽.

𝑣̂(𝑠,𝜽)≈𝑣𝜋(𝑠)

Our parameterised function could take many forms. For example, it could be a linear function, an artificial neural network, or a decision tree. No matter what type of function approximator we use, our goal is the same: **we want to learn a set of parameters for our function which allows it to map states to values correctly**.

# Learning from Example Data

Completion requirements

All of the prediction methods that we have discussed can be understood as moving existing estimates of a state's value towards a new estimate, or target value. We can denote such updates as 𝑠↦𝑣, where 𝑠is the state whose value is being updated and 𝑣 is the update that the value of state 𝑠 is being moved towards.

Depending on which method we are using, we have seen update targets computed in different ways. The update targets used by dynamic programming, Monte Carlo, and temporal-difference prediction methods are shown below.

|   |   |
|---|---|
|Dynamic Programming:|𝑠↦𝐸𝜋{𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)\|𝑆𝑡=𝑠}|
|Monte Carlo:|𝑠↦𝐺𝑡|
|Temporal-Difference:|𝑠↦𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)|

In the tabular case, our agent's estimate for the value of state 𝑠 is simply moved towards the target value 𝑣 (e.g. according to some learning rate 𝛼), and all other value estimates were left unchanged. In the function approximated case, things are not so simple. How, we need to find a way of changing our the parameters 𝜽 of 𝑣̂ so that the value of state 𝑠 is moved towards the target value 𝑣, in a way which generalises well to many other similar states.

To do this, we can take inspiration from supervised learning. Supervised learning methods use training examples — pairs of input and output values — to approximate functions. Imagine that our agent has just generated some experience and used it to compute a new estimate 𝑣 for the value of state 𝑠. We can use 𝑠 as an input and 𝑣 as an output in a supervised learning context in order to update the parameters of our agent's value function estimate.

Of course, using supervised learning methods in a reinforcement learning context does not come without its challenges. Supervised learning methods usually assume a fixed training dataset, whereas we will want to train our agent based on experience that it collects while interacting with its environment. We must also be able to deal with non-stationary target values, either due to our agent's policy changing over time or, if we are using bootstrapping methods, because our estimate itself is changing over time.

# Deriving an Update Rule

Completion requirements

In supervised learning, we usually evaluate how well a given set of parameters 𝜽 allows our function to correctly map inputs to outputs using an **objective function**.

A commonly used objective function is the **m****ean squared error (MSE)**. The MSE works by looking at 𝑚 pieces of input data for which we know the true outputs, and compares these correct outputs with the predicted outputs from our function. The output of the MSE is the square of the distance between our function's predicted output 𝑦̂ and the true output 𝑦, averaged across all 𝑚 pieces of data.

MSE=1𝑚∑𝑚𝑖=1(𝑦𝑖−𝑦̂𝑖)2

We commonly refer to the outputs of objective functions, such as the MSE, as the **loss**. In the case of MSE, a lower loss is desirable because it means that our function's outputs 𝑦̂ are very close to the true outputs 𝑦. So, it would make sense to perform updates to our parameters 𝜽 in a way that lowers the loss. We can quite easily perform such an update using gradient descent techniques.

Assuming that as our original function is differentiable, we can calculate the gradient of our objective function (e.g. MSE) with respect to our parameters 𝜽. This gradient tells us the direction we'd need to move our parameter values in to increase our loss, so moving them in the opposite direction will reduce it. Based on this, we can formulate the following simple update rule:

𝜽𝑖+1=𝜽𝑖−𝛼∇𝜽𝑖MSE

The figure below illustrates a simple instance of applying gradient descent to optimise a function with two parameters. The vertical axis shows the loss for various combinations of the two parameter values on the horizontal axes. Three sets of initial parameter values are shown, and the parameters are gradually adjusted until a local or global minimum of the loss function is reached.

**An illustration showing how gradient descent works.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512518&forceview=1#footnote-1)

To solve the prediction problem, we need to learn a function that correctly maps states to state values. It turns out we can easily adapt the MSE to serve as an objective function for learning a value function.

Recall that the MSE averages the squared distance between predicted outputs 𝑦̂, true outputs 𝑦 for given inputs:

MSE=1𝑚∑𝑚𝑖=1(𝑦𝑖−𝑦̂𝑖)2

Assume that we had just computed a new estimate for the value of a state, 𝑣𝜋(𝑠), based on our agent's experience (e.g. a sample return 𝐺𝑡). This will take the place of the 'true' output value 𝑦 in the MSE. We will compare this value to our agent's current estimate of the value of state 𝑠, 𝑣̂(𝑠,𝜽), which will replace 𝑦̂. For theoretical purposes, we'll also define a distribution over all states, 𝜇(𝑠), telling us how much we care about the error in each state.

VE⎯⎯⎯⎯⎯⎯⎯⎯=∑𝑠∈𝜇(𝑠)(𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽))2

We call this objective function the **mean squared value error** (VE⎯⎯⎯⎯⎯⎯⎯⎯). The intuition behind the VE⎯⎯⎯⎯⎯⎯⎯⎯ is the same as for the regular MSE, all it does is penalise differences between our agent's state-value prediction 𝑣̂(𝑠,𝜽) and the target value 𝑣𝜋(𝑠). The further our agent's estimate is away from the target value, the higher the (VE⎯⎯⎯⎯⎯⎯⎯⎯). Changing our agent's parameters 𝜽 to minimise the VE⎯⎯⎯⎯⎯⎯⎯⎯ would result in our agent's predicted state value moving closer to the target value.

In order to update our parameters 𝜽 in a way that minimises the loss, we need to define an update rule. As discussed above, we can use gradient descent techniques to perform updates. A gradient-descent update rule based on the VE⎯⎯⎯⎯⎯⎯⎯⎯ looks like the following:

𝜽𝑖+1=𝜽𝑖−12𝛼∇𝜽𝑖[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]2

Applying the chain rule ([𝑓(𝑔(𝑥))]′=𝑓′(𝑔(𝑥))𝑔′(𝑥)), we can simplify to get the following:

𝜽𝑖+1=𝜽𝑖+𝛼[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]∇𝜽𝑖𝑣̂(𝑠,𝜽𝑖)

This is the form of the update equation that you will see used most often. Once again, it's important to understand that the intuition here is the same as it always has been. This update rule tells us how to change our function's parameters in a way that moves our agent's prediction of the value of state 𝑠 towards the update target 𝑣𝜋(𝑠), thereby decreasing the loss.

# Learning from Example Data

Completion requirements

As mentioned previously, we can compute the target value we use for our updates in many ways. We could use a Monte Carlo return, a value computed using dynamic programming, or a temporal-difference estimate.

|   |   |
|---|---|
|Dynamic Programming:|𝑠↦𝐸𝜋{𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)\|𝑆𝑡=𝑠}|
|Monte Carlo:|𝑠↦𝐺𝑡|
|Temporal-Difference:|𝑠↦𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)|

There are some theoretical considerations to keep in mind when choosing which of these to use in our new gradient-descent update. Recall how we derived our gradient-descent update:

𝜽𝑖+1𝜽𝑖+1=𝜽𝑖−12𝛼∇𝜽𝑖[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]2=𝜽𝑖+𝛼[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]∇𝜽𝑖𝑣̂(𝑠,𝜽𝑖)

Here, we have assumed that our target value, 𝑣𝜋(𝑠), does not depend on our parameters 𝜽 in any way. For unbiased estimates which don't bootstrap, like the full returns used by Monte Carlo, this is true. However, bootstrapping estimates are inherently based on our agent's current predictions, so this is not true for them.

Intuitively, when using this update rule to update towards a bootstrapping estimate, the update takes into account the effect that changes to the parameters will have on our agent's estimate of the state 𝑠, but not the effect that changes to the parameters will have on the target it is updating towards.

We refer to methods that update towards unbiased estimates as **gradient descent methods**. We refer to methods that update towards biased estimates, **semi-gradient methods**. Despite more limited convergence guarantees being available for semi-gradient methods, they are still widely used and perform well in practice.

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Biased and Unbiased Estimates

Completion requirements

As mentioned previously, we can compute the target value we use for our updates in many ways. We could use a Monte Carlo return, a value computed using dynamic programming, or a temporal-difference estimate.

|   |   |
|---|---|
|Dynamic Programming:|𝑠↦𝐸𝜋{𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)\|𝑆𝑡=𝑠}|
|Monte Carlo:|𝑠↦𝐺𝑡|
|Temporal-Difference:|𝑠↦𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)|

There are some theoretical considerations to keep in mind when choosing which of these to use in our new gradient-descent update. Recall how we derived our gradient-descent update:

𝜽𝑖+1𝜽𝑖+1=𝜽𝑖−12𝛼∇𝜽𝑖[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]2=𝜽𝑖+𝛼[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]∇𝜽𝑖𝑣̂(𝑠,𝜽𝑖)

Here, we have assumed that our target value, 𝑣𝜋(𝑠), does not depend on our parameters 𝜽 in any way. For unbiased estimates which don't bootstrap, like the full returns used by Monte Carlo, this is true. However, bootstrapping estimates are inherently based on our agent's current predictions, so this is not true for them.

Intuitively, when using this update rule to update towards a bootstrapping estimate, the update takes into account the effect that changes to the parameters will have on our agent's estimate of the state 𝑠, but not the effect that changes to the parameters will have on the target it is updating towards.

We refer to methods that update towards unbiased estimates as **gradient descent methods**. We refer to methods that update towards biased estimates as **semi-gradient methods**. Despite more limited convergence guarantees being available for semi-gradient methods, they are still widely used and perform well in practice.

# Semi-Gradient TD(0)

Completion requirements

Now that we have derived an update rule that we can use to update our agent's parameters whenever it computes new estimates for the values of states, we are now ready to look at a function-approximated prediction algorithm called **s****emi-gradient TD(0)**.

Recall that tabular TD(0) performed a one-step TD update each time-step. After transitioning to state 𝑆′and earning reward 𝑅 after taking action 𝐴 in state 𝑆, tabular TD(0) would perform the following update:

𝑉(𝑆)←𝑉(𝑆)+𝛼[𝑅+𝛾𝑉(𝑆′)−𝑉(𝑆)]

Remember, the intuition here is that our agent's estimate of the value of state 𝑆 is being moved away from its current value, 𝑉(𝑆), towards a target value, the TD target 𝑅+𝛾𝑉(𝑆′).

After observing the same experience, semi-gradient TD(0) performs a similar update, based on the generic gradient-descent update rule we derived previously. Instead of updating the value of 𝑆 in a table, we update our agent's parameters using the following update rule:

𝜽=𝜽+𝛼[𝑅+𝛾𝑣̂(𝑆′,𝜽)−𝑣̂(𝑠,𝜽)]∇𝜽𝑣̂(𝑠,𝜽)

The intuition behind this update is the same as before: we are changing our parameters so that our agent's value function estimate, 𝑣̂(𝑆,𝜽), moves closer to the TD target, 𝑅+𝛾𝑣̂(𝑆′,𝜽). The difference is that this update will affect the estimated value of many states, instead of just 𝑆. The pseudocode for the semi-gradient TD(0) algorithm can be found below.

Semi-gradient T D(0) for estimating 𝑣̂≈𝑣𝜋

 Input: a policy 𝜋 to be evaluated  Input: a differentiable function 𝑣̂:×ℝ𝑑→ℝ where 𝑣̂( terminal, ⋅)=0 Algorithm parameter: learning rate 𝛼∈(0,1] Initialise value-function parameters 𝜃∈ℝ𝑑 arbitrarily (e.g. 𝜃=0 )  Loop for each episode:  Initialise 𝑆 Loop for each step of episode: 𝐴← action chosen in state 𝑆 by policy 𝜋 Take action 𝐴, observe reward 𝑅 and next-state 𝑆′𝜽←𝜽+𝛼[𝑅+𝛾𝑣̂(𝑆′,𝜽)−𝑣̂(𝑆,𝜽)]∇𝜽𝑣̂(𝑆,𝜽)𝑆←𝑆′

**Pseudocode for the semi-gradient TD(0) algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512521&forceview=1#footnote-1)

Notice how the overall structure of this algorithm has not changed from its tabular version. The main differences can be found in the initialisation step, where we must now initialise the parameters of a function instead of entries in a table, and in the update step, where we now use our new semi-gradient update rule instead of a tabular update rule.

# Semi-Gradient Sarsa

Completion requirements

We will finish this lesson by extending function approximation to solve the control problem. We have already seen semi-gradient TD(0) for solving the prediction problem, where our goal was to learn a function 𝑣̂ with parameters 𝜽 which maps states to state-values. Would it be difficult to come up with a similar method for solving the control problem, where our goal is to learn a function 𝑞̂ with parameters 𝜽which maps state-action pairs to action-values?

As it turns out, it is actually very straightforward to extend the ideas we have already used to solve the prediction problem to solving the control problem. Just as we have seen many methods so far in this unit for computing state-value estimates, we have also seen many methods for computing action-value estimates.

|   |   |
|---|---|
|Monte Carlo:|𝑠↦𝐺𝑡|
|Temporal-Difference:|𝑠↦𝑅𝑡+1+𝛾𝑣̂(𝑆𝑡+1,𝜽𝑡)|

Similarly to what we did before, we can use state-action pairs 𝑠,𝑎 and update targets computed by our agent as training examples in a supervised learning context. Really, there is no fundamental difference between approximating a state-value function and an action-value function.

Previously, we derived the following update rule for updating the parameters of our agent's state-value function estimate:

𝜽𝑖+1=𝜽𝑖+𝛼[𝑣𝜋(𝑠)−𝑣̂(𝑠,𝜽𝑖)]∇𝜽𝑖𝑣̂(𝑠,𝜽𝑖)

We can use a very similar update rule for updating the parameters of an action-value function:

𝜽𝑖+1=𝜽𝑖+𝛼[𝑞𝜋(𝑠,𝑎)−𝑞̂(𝑠,𝑎,𝜽𝑖)]∇𝜽𝑖𝑞̂(𝑠,𝑎,𝜽𝑖)

We are now ready to look at a function-approximated TD control algorithm called \textbf{Semi-Gradient Sarsa}. Recall that, after observing a transition to state 𝑆′ and reward 𝑅 after taking action 𝐴 in state 𝑆, tabular Sarsa would perform the following update:

𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾𝑄(𝑆′,𝐴′)−𝑄(𝑆,𝐴)]𝐴′ chosen according to 𝜋

Using the TD target 𝑅+𝛾𝑞̂(𝑆′,𝐴′,𝜽), semi-gradient Sarsa would perform the following update:

𝜽=𝜽+𝛼[𝑅+𝛾𝑞̂(𝑆′,𝐴′,𝜽)−𝑞̂(𝑆,𝐴,𝜽𝑖)]∇𝜽𝑖𝑞̂(𝑆,𝐴,𝜽𝑖)

The intuition behind this update rule is the same as always. We want to move our parameters in a direction that minimises the loss. In our case, this means moving them in a direction that moves the function's output away from its current value and towards the TD target. The full pseudocode for semi-gradient Sarsa can be found below.

Semi-gradient Sarsa for estimating 𝑞̂≈𝑞∗

 Input: a differentiable function 𝑞̂:××ℝ𝑑→ℝ where 𝑞̂( terminal ,⋅,⋅)=0 Algorithm parameters: learning rate 𝛼>0, small exploration rate 𝜖>0 Initialise value-function parameters 𝜃∈ℝ𝑑 arbitrarily ( e.g. 𝜃=0) Loop for each episode:  Initialise 𝑆 Choose action 𝐴 in state 𝑆 using policy derived from 𝑞̂(𝑆,⋅,𝜃) Loop for each step of episode: Take action 𝐴, observe reward 𝑅 and next-state 𝑆′Choose action 𝐴′ in state 𝑆′ using policy derived from 𝑞̂(𝑆,⋅,𝜃)𝜃←𝜃+𝛼[𝑅+𝛾𝑞̂(𝑆′,𝐴′,𝜃)−𝑞̂(𝑆,𝐴,𝜃)]∇𝜃𝑞̂(𝑆,𝐴,𝜃)𝑆←𝑆′𝐴←𝐴′

**Pseudocode for the semi-gradient Sarsa algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512522&forceview=1#footnote-1)

# Lesson 2 Summary

Completion requirements

In this lesson, we have introduced **function approximation** in reinforcement learning.

We have looked at how we can use example data generated by our agent as it interacts with its environment to train state-value and action-value functions using gradient descent methods from supervised learning. We derived gradient descent update rules and examined function-approximated TD methods for solving the prediction and control problems. We have discussed:

- that we can **represent value functions as parameterised functions** that map states/state-action pairs to values, instead of storing each value separately in a table,
- that, by **updating our function's parameters**, we can change the estimated value of many states/state-action pairs with a single update,
- that we can use **gradient descent techniques** from supervised learning to update our function's parameters in response to new experience,
- how to implement the **s****emi-gradient TD(0) algorithm**, and
- how to implement the **semi-gradient Sarsa algorithm**.
# Lesson 3 Introduction

Completion requirements

So far this week, we have kept things quite abstract. We have spoken at length about using parameterised functions to represent our value functions, but have not yet specified what form our function and its parameters should take. We will now look at the concrete case of using a linear function as our function approximator.

Linear functions are relatively simple to understand, which makes them a useful type of function approximation to consider first. They are also easy to analyse mathematically, and many convergence proofs for function-approximated reinforcement learning methods hold only for the linear function-approximation case. At the end of this lesson, we will look at applying the semi-gradient Sarsa algorithm to the simple Mountain Car problem using linear function approximation.

# Linear Function Approximation

Completion requirements

We will now look at the special case of using a **linear function** to represent our approximate value function. Recall that a linear function is simply a weighted sum of its inputs, and its parameters are the weights to be applied to each term. We will discuss state-value functions here, but remember that we can apply these ideas to action-value functions just as easily.

#### **Linear functions as function approximators**

---

Imagine that, for every state 𝑠 in our state-space, we have some real vector of length 𝑑 to represent it. We refer to this vector as the **feature vector** representing each state, denoted 𝐱(𝑠)∈ℝ𝑑. Essentially, we have a vector of 𝑑 numbers representing each state. This feature vector will serve as the input to our linear function.

Our function's parameter vector 𝜽∈ℝ𝑑 is going to contain the weights applied to each term of the feature vector. Notice that the feature vector and the parameter vector are the same length, 𝑑.

So, for some state 𝑠, the output of our linear function 𝑣̂(𝑠,𝜽) is going to be the sum of the elements of our feature vector 𝐱(𝑠), weighted by the elements of our function's parameters 𝜽.

𝑣̂(𝑠,𝜽)=∑𝑑𝑖=0𝜃𝑖x𝑖(𝑠)=𝜽⊺𝐱(𝑠)

We now have a very simple route from some original representation of a state to our approximated value for that state. Based on the original state representation, we derive our feature vector. We use the feature vector as input to our linear function, multiplying each element by its corresponding weight in our parameter vector. This gives us some real-valued output, which is our function's value estimate.

Original Representation→Feature Vector→Linear Function→Approximated Value

#### **Benefits of linear function approximation**

---

There are several benefits to using linear functions as our function approximators. Firstly, because a linear function is simply a linear sum of terms, it is easy to compute the gradient with respect to our parameters.

𝑣̂(𝑠,𝜽)∇𝜽𝑣̂(𝑠,𝜽)=∑𝑖=0𝑑𝜃𝑖𝑥𝑖(𝑠)=⎡⎣⎢⎢⎢⎢𝑥0(𝑠)𝑥1(𝑠)⋮𝑥𝑑(𝑠)⎤⎦⎥⎥⎥⎥

This makes performing gradient descent with linear methods rather simple and easy to analyse mathematically. In fact, the majority of convergence guarantees for function-approximated reinforcement learning methods assume linear function approximation. If we switch to more complex function approximators, such as artificial neural networks, the convergence guarantees are weaker.

# Coarse Coding

Completion requirements

Before applying linear function approximation methods, we need a way to derive feature vectors that represent states in our agent's environment. This is the problem of **feature construction**. Extracting useful features appropriate to a given problem is a common challenge in machine learning. The elements of our feature vector should capture aspects of our agent's environment that are useful for generalising across. We will look at a feature-extraction method called **coarse coding** and apply it to the classic Mountain Car environment.

Recall that in the Mountain Car environment, our agent must drive a car from its initial position between two hills to a goal on the hill to its right as quickly as possible. Our agent can drive full throttle forward, full throttle backwards, or set the throttle to zero. However, the car is underpowered and cannot drive all the way up the hill on its own. To reach the goal, the agent must drive up the left hill and use gravity to build enough momentum to reach the top of the right hill. This problem is illustrated below.

**An illustration of the Mountain Car problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1512528&forceview=1#footnote-1)

States in this environment are represented by two continuous variables: the car's position on the 𝑥-axis (−1.2<𝑥<0.5) and its velocity (−0.07<𝑣<0.07). Our agent earns a reward of −1 per time step, and the episode ends when it reaches the goal.

Coarse coding is a simple method of generating feature vectors for environments with continuous state spaces. Because the Mountain Car environment's state space has only two dimensions, we can represent it as a rectangle. Let us arrange a set of circles to randomly cover the entire state-space, as shown in the image below. For clarity, circles near the edge have been omitted.

**Coarse coding applied to the state-space of the Mountain Car problem. Here, large circles have been used, leading to broad generalisation between states.**[Moodle - Image Description](https://moodle.bath.ac.uk/mod/page/view.php?id=1512528&forceview=1#footnote-2) 

Coarse coding works as follows. First, we add an entry for each circle to our feature vector. If a given state lies inside one of the circles, we set that circle's entry in our feature vector to 1. Otherwise, we set it to 0. We call such features **binary features**. In the figure shown above, an example state has been marked with a cross. In this marked state's feature vector, the entries for the three shaded circles would be set to 1, and the entries for all the other circles would be set to 0. This feature vector can then be used as the input for a linear function (or other function approximator).

You may be wondering how this helps with generalisation. If we learn about a state that lies within one circle, some of what we learn about that state will be applied to all of the other states that lie within that circle. The more circles that two different states have in common, the more strongly the generalisation between them. In fact, if two states share the exact same set of circles, they will effectively be treated as identical. Conversely, if two states don't share any circles in common, there will be no immediate generalisation between them.

Of course, we can control the generalisation by changing the size and shape of our circles. If we used smaller circles, as shown in the image below, information about a state will be spread to a state less far away than for the larger circles shown above.

**Coarse coding applied to the state-space of the Mountain Car problem. Here, large circles have been used, leading to broad generalisation between states.**[Moodle - Image Description](https://moodle.bath.ac.uk/mod/page/view.php?id=1512528&forceview=1#footnote-3) 

Similarly, if we knew that we wanted broad generalisation along one state variable, but narrow generalisation along the other one, we could use ovals instead of perfect circles, as shown in the image below.

**Coarse coding applied to the state-space of the Mountain Car problem. Here, large circles have been used, leading to broad generalisation between states.**[Moodle - Image Description](https://moodle.bath.ac.uk/mod/page/view.php?id=1512528&forceview=1#footnote-4) 

In summary, coarse coding is a simple way to generalise across similar states when working with continuous state spaces. It lets us spread what our agent learns about one state to other nearby states.

# Tile Coding

Completion requirements

In practice, it is usually more convenient to implement coarse coding using grids instead of circles. This simplifies many distance calculations and makes it easier to ensure full coverage of the state space. Square tiles also scale nicely to higher dimensions (e.g., cubes for 3D state spaces). 

Implementing coarse coding in this way is called **tile coding**.

In the image below, you can see tile coding with a single grid. The grid is split into a series of tiles, and each tile would have an entry in the feature vector. If a given state is inside a tile, its entry is set to 1. Otherwise, it is set to 0.

**Tile coding with a single grid applied to a continuous two-dimensional state-space.**

Of course, using a single grid of tiles means that we do not have any overlapping tiles, which hinders generalisation. To solve this, we can use multiple grids of tiles where each grid is randomly offset, as shown in the image below. As long as every grid covers the entire state-space, we will end up with each state being within one tile from each grid.

**Tile coding with three grids applied to a continuous two-dimensional state-space.**

Additionally, we are also free to alter the size and shape of our grid tiles, as long as the entire state-space remains covered. For instance, we might use rectangles instead of squares to generalise more broadly across one dimension than another, or use stripes to generalise across an entire dimension, as shown in the image below. As long as the tiling covers the entire state-space and is simple to compute, we retain all the benefits of square tiles while importing domain knowledge to help ensure better generalisation.

**Alternative tiling shapes.**

Finally, an interesting thing to consider is the effect that the offsets between our tiles grids have on generalisation. Consider the image from the course textbook shown below. If we use uniform offsets for our tile grids, we might end up with more generalisation along one direction than others, which might not be desirable. Conversely, if we offset each of our tile grids randomly, then we end up with much more natural-looking generalisation.

# Lesson 3 Summary

Completion requirements

In this lesson, we have discussed using linear function approximation in reinforcement learning. We have looked at how to use linear functions to approximate value functions and generated feature vector inputs for them using coarse coding. We ended by looking at the performance of semi-gradient Sarsa on the Mountain Car problem, using linear function approximation and tile coding. We have discussed

- how **linear functions** can be used to approximate value functions in reinforcement learning,
- some of the benefits of using **linear function approximation**,
- how to use **coarse coding** to perform **feature extraction**, and
- how to implement a variant of coarse coding called **tile coding**.

# Weekly Summary

Completion requirements

This week, we have introduced the concept of function approximation in reinforcement learning. We started by discussing how the tabular reinforcement learning methods that we have studied so far do not scale well to more complex sequential decision problems with large state spaces. This is partly due to their memory requirements scaling with the number of states in the state space, and more significantly, because our agent cannot expect to visit some (or even most) states. Because tabular methods learn and store values separately, they do not support the generalisation required for our agent to perform well in such large environments.

We then introduced function approximation as a potential solution to this problem. Instead of representing our value functions as tables, we represented them as parameterised functions. Instead of learning and storing values separately, function approximation allowed us to learn the parameters of a function that mapped states to values. Changing a single parameter can affect many states, enabling generalisation, and storing a small set of parameters uses much less memory than storing values for every state.

We discussed using ideas from supervised learning methods to learn approximate value functions, treating state inputs and estimated value outputs as training examples. We then derived a gradient descent update rule for our function's parameters, with which it can use its experience to better map states to values. We then used this update rule to implement semi-gradient TD(0), a function-approximated TD prediction algorithm. We then discussed how easy it was to apply these same ideas to learning action-values, and implemented a function-approximated version of the Sarsa control algorithm.

Finally, we considered the special case of using linear functions as our approximators. Linear functions take a feature vector representing a state or state-action pair as input and return a weighted sum of this vector's elements as their value estimate output. We used coarse coding and, later, tile coding to extract features in continuous problems, and looked at the results of applying semi-gradient Sarsa with linear function approximation to the Mountain Car problem.