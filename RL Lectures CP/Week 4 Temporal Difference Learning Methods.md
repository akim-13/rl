# Lesson 1 Introduction

Completion requirements

In this lesson, we will introduce a third class of solution methods for sequential decision problems: temporal-difference methods. As we will discuss, the temporal-difference learning is, arguably, one of the most novel and significant ideas unique to reinforcement learning.

Like Monte Carlo methods, temporal-difference methods do not rely on access to a perfect model of our agent's environment, and learn directly from our agent's experiences. However, unlike Monte Carlo methods, temporal-difference methods do not require full episodes of experience in order to learn. Instead, they use bootstrapping, like dynamic programming methods, to learn from shorter fragments of experience.

In this lesson, we will be looking at how temporal-difference methods can be used for policy evaluation, and discussing how they compare with Monte Carlo and dynamic programming methods.

# Temporal-Difference Updates

Completion requirements

Perhaps the best way to introduce **t****emporal-difference (TD)** methods is to compare the backups they perform with those performed by the other two classes of solution method that we've already looked at. As usual, we're going to consider the policy evaluation (or, prediction) problem initially, before moving on to the control problem later.

Recall that, if we have access to a perfect model of our agent's environment, we can apply dynamic programming methods to solve that problem. To estimate the value of a given state, dynamic programming methods use the model to perform a one-step look-ahead at all of the possible immediate rewards and next-states that can be reached from that state, then perform an update based on the Bellman equations.

Recall how dynamic programming methods use the concept of **_bootstrapping_**: the updated value estimate for one state will be based on existing value estimates of its possible successors.

𝑉(𝑠)=∑𝑎𝜋(𝑎|𝑠)∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑉(𝑠′)]

**The backup diagram associated with the Bellman equation for 𝑣𝜋.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502486&forceview=1#footnote-1)  
  

We then introduced Monte Carlo methods, which learn using experience data _**sampled**_ by the agent through interaction with the environment, without relying on any prior knowledge whatsoever. Using full episodes of experience, Monte Carlo methods compute sample returns earned after visiting each state, and form value estimates by averaging these returns.

Because these value estimates are formed independently and do not rely on each other, it is easy to see that Monte Carlo methods do not bootstrap.

𝑉(𝑠)=average(Returns(𝑠))

**A backup diagram representing a Monte Carlo backup for estimating 𝑣𝜋.**  [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502486&forceview=1#footnote-2)

The methods that we will look at this week, temporal-difference learning methods, give us another alternative. Like dynamic programming methods, temporal-difference learning methods look only a single step ahead when updating the value of a state, and bootstrap off of existing value estimate for that state's successors. This allows temporal-difference learning methods to learn from shorter fragments of experience than full episodes. Alongside this, temporal-difference learning methods learn using an agent's experience, sampling experience data like Monte Carlo methods do, instead of relying on information from a model like dynamic programming methods.

The backup diagram for a temporal-difference learning method estimating 𝑣𝜋 is shown below. Later in this lesson, we are going to look at a temporal-difference prediction algorithm which implements the type of update illustrated in this backup diagram.


**A backup diagram representing a temporal-difference backup for estimating 𝑣𝜋.**   [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502486&forceview=1#footnote-3)

Based on the descriptions of each class of solution method given above, we can identify two key dimensions in which they differ. The first is whether or not they bootstrap: whether they base their new value estimates on existing ones. The second is whether or not they sample: whether they learn using experience sampled by our agent or rely on a model of the environment.

These differences are summarised in the table below.

||Dynamic Programming|Monte Carlo|Temporal-Difference|
|---|---|---|---|
|Bootstrapping|Yes|No|Yes|
|Sampling|No|Yes|Yes|


![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Incremental Updates

Completion requirements

Previously, when working with Monte Carlo methods, we computed value estimates by collecting and averaging lists of sample returns.

To avoid having to store and average ever-growing lists of sample returns, it would be better to store a single value for each state or state-action pair and update it incrementally. For some problems, such as those where the environment's dynamics change over time, it also makes sense to prioritise recent experiences over older ones.

Let us say that we currently hold some estimate of the value of a state and then compute some new estimate, which we will call our **target**. A simple incremental update would take a step away from the existing estimate and towards the target value. The size of the step we take towards the target is controlled using a **step-size** or **learning rate** parameter 0≥𝛼≥1, and our updated value estimate becomes a weighted sum of our existing estimate and our new estimate.

Suppose that a Monte Carlo method holds some current estimate of a state 𝑆𝑡, 𝑉(𝑆𝑡, and observes a new sample return 𝐺𝑡. A simple incremental update would look like:

𝑉(𝑆𝑡)←(1−𝛼)𝑉(𝑆𝑡)+𝛼𝐺𝑡

We call the Monte Carlo method that uses this update **constant-𝛼 Monte Carlo prediction**.

Remember, the intuition behind this update is simple: we move away from our existing estimate and towards the new sample return we have just observed.

As shown below, we can also rearrange this equation into a different form. This is the form of incremental update that we will see used in pseudocode most often.

𝑉(𝑆𝑡)𝑉(𝑆𝑡)𝑉(𝑆𝑡)←(1−𝛼)𝑉(𝑆𝑡)+𝛼𝐺𝑡←𝑉(𝑆𝑡)−𝛼𝑉(𝑆𝑡)+𝛼𝐺𝑡←𝑉(𝑆𝑡)+𝛼[𝐺𝑡−𝑉(𝑆𝑡)]


# TD(0) Policy Evaluation

Completion requirements

We are now ready to look at **TD(0)**, the simplest temporal-difference prediction algorithm.

For an agent that starts in state 𝑆𝑡, takes action 𝐴𝑡 according to its policy 𝜋, transitions to state 𝑆𝑡+1and earns reward 𝑅𝑡+1, TD(0) performs the following update:

𝑉(𝑆𝑡)←𝑉(𝑆𝑡)+𝛼[𝑅𝑡+1+𝛾𝑉(𝑆𝑡+1)−𝑉(𝑆𝑡)]

In this incremental update, TD(0) moves away from its existing state-value estimate 𝑉(𝑆𝑡) and towards the target value 𝑅𝑡+1+𝛾𝑉(𝑆𝑡+1). This target value incorporates new information about the environment that our agent has just observed: the immediate reward 𝑅𝑡+1, and bootstrapping off of its estimate value of the next-state, 𝛾𝑉(𝑆𝑡+1).

We can clearly see that this temporal-difference update combines the sampling performed by Monte Carlo methods with the bootstrapping performed by dynamic programming methods. Notice that this temporal-difference update is clearly derived from the Bellman equation for 𝑣𝜋. The full pseudocode for the TD(0) algorithm is shown below.

Tabular TD(0) for estimating 𝑣𝜋

Input: the policy 𝜋 to be evaluated Algorithm parameter: step size 𝛼∈(0,1]Initialize 𝑉(𝑠), for all 𝑠∈+,arbitrarily except that 𝑉( terminal )=0Loop for each episode: Initialize 𝑆Loop for each step of episode: 𝐴←action given by 𝜋 for 𝑆Take action 𝐴, observe 𝑅,𝑆′𝑉(𝑆)←𝑉(𝑆)+𝛼[𝑅+𝛾𝑉(𝑆′)−𝑉(𝑆)]𝑆←𝑆′ until 𝑆 is terminal 

**Pseudocode for the TD(0) algorithm.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502488&forceview=1#footnote-1)

An important thing to notice when comparing this prediction method with the first-visit Monte Carlo prediction algorithm we saw previously is that **TD(0)** can perform updates to its value function every single time-step, whereas the Monte Carlo method had to wait until the end of each episode before it could perform any updates. The reason that our TD(0) can perform these **one-step** updates is because it makes use of bootstrapping. The TD update doesn't rely on the full list of returns up until the end of an episode, only the immediate reward and some estimate for the value of the next-state.

# Advantages of TD Learning

Completion requirements

---

### Stop and Think



Let's stop for a moment to consider the advantages that TD methods offer.

- TD methods learn directly from an agent's experience, and do not require a model of the environment.
- TD methods can be fully incremental, allowing agents to learn from fragments of experience as short as a single time-step.
- TD methods can learn before reaching a goal-state, or even without ever reaching a goal-state at all.

Take a moment to think about which (if any) of these statements can also be applied to dynamic programming methods or Monte Carlo methods.

---

#### **Optimality of TD and MC Methods**

There is no clear theoretical answer as to whether TD methods are superior to MC methods, but empirical evidence often suggests that this is the case. We will spend the rest of this lesson comparing TD prediction methods with MC prediction methods to gain a deeper understanding of the differences between them.

We will be leaning heavily on this week's required reading material for the examples that follow. So, please forgive the sizable overlap in content, and please refer to the book for more details regarding the examples we are about to discuss.

Consider the environment illustrated below, which consists of a chain of seven states. Actions are available which allow movement between adjacent states, and the states at each end of the chain are terminal states. All rewards in this environment are zero, except for transitioning to the rightmost terminal state.

**Illustration of the random walk problem.**  [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502489&forceview=1#footnote-1)

Let us pose the problem of evaluating the equiprobable random policy in this environment. The graph below shows the true values of each of the non-terminal states, as well as the values estimated by TD(0) after 1, 10, and 100 episodes with 𝛼=0.1.

**State-value estimates made by TD(0) on the random walk problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502489&forceview=1#footnote-1)

You may be wondering why, even after 100 episodes, TD(0)'s estimate is nearing the true values but has not converged perfectly. TD(0) is only guaranteed to converge if a gradually-decreasing value of 𝛼 is used. However, these results should help convince you that TD(0) produces reasonable estimates when performing policy evaluation, at least in this simple domain.

The results of performing 100 runs of policy evaluation using TD(0) and constant-𝛼 Monte Carlo prediction on this problem using various values of 𝛼 are shown in the graph below. This graph shows the root-mean-square error between the estimated values of each state and their true values, for each method and value of 𝛼.

**Results showing the error in value estimates made by TD(0) and constant-𝛼MC prediction on the random walk problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502489&forceview=1#footnote-1)

From looking at this graph, we can see quite clearly that, for various values of 𝛼, TD(0) consistently produces better state-value estimates than constant-𝛼 Monte Carlo prediction. On top of this, we can also see that TD(0) is learning at a faster rate than constant-𝛼 Monte Carlo prediction in all cases.

To understand why TD(0) consistently outperforms constant-𝛼 Monte Carlo prediction on the random walk problem, we will take a look at another example. For this example, we are going to assume that our algorithms are performing **batch updates**. When performing batch updates, we assume a fixed set of experiences that is repeatedly run through an algorithm until it converges upon an answer. Value estimates are also only changed after completing each full pass through the fixed set of experiences. This is not something that we would often do in practice, but it is useful in this case where we are trying to analyse how two different algorithms behave.

There is a theoretical guarantee that, for any Markov prediction task, under batch updating, TD(0) will converge for a sufficiently small value of 𝛼. Constant-𝛼 MC prediction is also guaranteed to converge under these conditions, **but to a different answer**. It is the difference between TD(0)'s answer and constant-𝛼 MC prediction which explains, at least in part, the poor performance of the latter on the random walk problem.

Suppose that our agent observed the following eight episodes of interaction while solving an unknown undiscounted sequential decision problem.

|𝐴,0,𝐵,0|𝐵,1|
|---|---|
|𝐵,1|𝐵,1|
|𝐵,1|𝐵,1|
|𝐵,1|𝐵,0|

**Eight episodes of interaction observed by our agent in this unknown environment.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502489&forceview=1#footnote-1)

What would be sensible estimates for the values of states 𝐴 and 𝐵? The answer seems clear for 𝐵: six out of eight of the returns earned after visiting it were 1, so its value should be 0.75. In fact, this is the answer that both TD(0) and constant-𝛼 MC prediction arrive at.

Deciding upon a value for state 𝐴, however, is not so easy. On one hand, all of the returns generated after visiting state 𝐴 are 0, so it is arguable that a good estimate of its value would be 0; this is the answer that constant-𝛼 MC prediction arrives at.

On the other hand, we can also take into account the fact that this is a sequential decision problem. Using the experiences generated by our agent, we can produce a model of our agent's environment, as illustrated below.

**A model of the unknown environment derived from the eight episodes of experience observed by our agent.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502489&forceview=1#footnote-1)

Using this model, we can infer that the value of state 𝐴 is composed of the immediate reward observed from transitioning to state 𝐵, plus the value of state 𝐵 itself. So, an alternative value for state 𝐴 would be 0+0.75=0.75; this is the answer that TD(0) arrives at.

This is the key difference between the answers reached by TD and MC methods. The MC value estimate minimises the mean-squared error on the training set of experiences, but it does not take into account the sequentiality of the problem. The TD value estimate is known as a **certainty-equivalence estimate**, and takes into account the sequentiality of the problem. It is the same estimate that we'd get if we constructed a model of our agent's environment using its experiences and used that model to compute value estimates.

# Lesson 1 Summary

Completion requirements

In this lesson, we have introduced **temporal-difference (TD)** learning methods. These methods combine many of the advantages of the dynamic programming and Monte Carlo methods that we have seen previously: they learn using an agent's experiences without relying on a perfect world model, while also being able to learn from fragments of experience shorter than full episodes. We have discussed

- how the updates performed by temporal-difference learning methods differ from the updates performed by dynamic programming and Monte Carlo methods,
- that temporal-difference learning methods make use of both **bootstrapping** and **sampling**,
- how to perform **incremental updates** to value estimates, instead of storing long lists of returns,
- how to implement the **TD(0)** algorithm,
- how temporal-difference learning methods can be seen to outperform Monte Carlo empirically, and
- how the **certainty-equivalence estimates** reached using temporal-difference learning methods are more accurate value estimates than those reached by the MC method because they take into account the sequentiality of sequential decision problems.
# Lesson 2 Introduction

Completion requirements

Now that we've seen a simple temporal-difference learning method for solving the prediction problem, we will move on to solving the control problem: how to learn optimal policies. The two temporal-difference control algorithms we're going to look at are **Sarsa**, an on-policy method, and **Q-Learning**, an off-policy method. These two algorithms are extremely important, and almost all of the algorithms that we'll consider later in the unit will be based on them.

# Sarsa: On-Policy TD Control

Completion requirements

Just as we previously discussed in the case of Monte Carlo methods, it is possible to extend temporal-difference learning methods to solving the control problem by estimating action-values instead of state-values. We can quite easily come up with an update equation for action-values that is very similar to the one we saw used in TD(0) to estimate state-values.

After observing our agent transition from state 𝑆𝑡 to next-state 𝑆𝑡+1 via action 𝐴𝑡 and earning reward 𝑅𝑡+1, we previously saw TD(0) perform the following update to its state-value estimate 𝑉(𝑆𝑡):

𝑉(𝑆𝑡)←𝑉(𝑆𝑡)+𝛼[𝑅𝑡+1+𝛾𝑉(𝑆𝑡+1)−𝑉(𝑆𝑡)]

Using the same experience, we can perform the following update to the action-value estimate: 𝑄(𝑆𝑡,𝐴𝑡):

𝑄(𝑆𝑡,𝐴𝑡)←𝑄(𝑆𝑡,𝐴𝑡)+𝛼[𝑅𝑡+1+𝛾𝑄(𝑆𝑡+1,𝐴𝑡+1)−𝑄(𝑆𝑡,𝐴𝑡)]

Remember that the intuition here is exactly the same as we saw when moving from policy evaluation to control last week: we're just learning about the values of state-action pairs instead of the values of states.

Doing this allows us to compare the actions available in each state without needing to perform a one-step look-ahead, which is necessary because temporal-difference learning methods do not assume access to a perfect model of the world.

**Illustration of generalised policy iteration (GPI) using an action-value function.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502493&forceview=1#footnote-1)

**Sarsa** is an on-policy TD control algorithm and is yet another example of GPI, where a policy evaluation and policy improvement process interact in order to learn an optimal policy. Each time-step, we perform a one-step update to our action-value function estimate based on what our agent has experienced, and derive our agent's policy from this action-value function estimate (using an 𝜖-greedy approach, for example).

The full pseudocode for the Sarsa algorithm is shown below.

Sarsa (on-policy TD control) for estimating 𝑄≈𝑞∗

Algorithm parameters: step size 𝛼∈(0,1], small 𝜀>0Initialize 𝑄(𝑠,𝑎), for all 𝑠∈+,𝑎∈(𝑠), arbitrarily except that 𝑄( terminal ,⋅)=0Loop for each episode: Initialize 𝑆Choose 𝐴 from 𝑆 using policy derived from 𝑄 (e.g., 𝜀-greedy) Loop for each step of episode: Take action 𝐴, observe 𝑅,𝑆′Choose 𝐴′ from 𝑆′ using policy derived from 𝑄 (e.g., 𝜀-greedy)  𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾𝑄(𝑆′,𝐴′)−𝑄(𝑆,𝐴)] 𝑆←𝑆′;𝐴←𝐴′;until 𝑆 is terminal 

**Pseudocode for the Sarsa algorithm.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502493&forceview=1#footnote-2)

Lets explore the pseudocode below.

An interesting detail is that the name “Sarsa” is an acronym derived from the five pieces of information that this algorithm incorporates into its updates: (𝑆𝑡,𝐴𝑡,𝑅𝑡+1,𝑆𝑡+1,𝐴𝑡+1).

# Q-Learning: Off-Policy TD Control

Completion requirements

We are now going to look at an off-policy TD control method called Q-Learning. Recall that off-policy methods learn about a target policy 𝜋 using experience generated using another behaviour policy 𝑏.

Q-Learning learns about the optimal policy directly using experience generated by, for instance, an 𝜖-greedy policy.

Recall the update rule used by the Sarsa algorithm:

𝑄(𝑆𝑡,𝐴𝑡)←𝑄(𝑆𝑡,𝐴𝑡)+𝛼[𝑅𝑡+1+𝛾𝑄(𝑆𝑡+1,𝐴𝑡+1)−𝑄(𝑆𝑡,𝐴𝑡)]

Q-Learning uses the following update rule:

𝑄(𝑆𝑡,𝐴𝑡)←𝑄(𝑆𝑡,𝐴𝑡)+𝛼[𝑅𝑡+1+𝛾max𝑎𝑄(𝑆𝑡+1,𝑎)−𝑄(𝑆𝑡,𝐴𝑡)]

Notice that Q-learning's update rule is almost identical to Sarsa's update rule except for one important detail.

- Sarsa bootstraps off of the value of the action that our agent would choose in the next-state.
- Q-Learning bootstraps off of the value of the highest-valued action available in the next-state.

No matter what behaviour policy 𝑏 is used to generate experience, Q-learning is able to use it to learn about the policy which always chooses the best available action: the optimal policy.

The backup diagrams for these two algorithms are illustrated below for comparison.

**Backup diagrams for the update rules used in the Sarsa and Q-learning algorithms.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502494&forceview=1#footnote-1)

The full pseudocode for Q-learning is shown below. The only significant difference between this pseudocode and the pseudocode that we examined earlier for Sarsa is the modified update rule.

Q-learning (off-policy TD control) for estimating 𝜋≈𝜋∗

Algorithm parameters: step size 𝛼∈(0,1], small 𝜀>0Initialize 𝑄(𝑠,𝑎), for all 𝑠∈+,𝑎∈(𝑠),arbitrarily except that 𝑄( terminal ,⋅)=0Loop for each episode: Initialize 𝑆Loop for each step of episode: Choose 𝐴 from 𝑆 using policy derived from 𝑄( e.g., 𝜀-greedy )Take action 𝐴, observe 𝑅,𝑆′𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾max𝑎𝑄(𝑆′,𝑎)−𝑄(𝑆,𝐴)]𝑆←𝑆′until 𝑆 is terminal 

**Pseudocode for the Q-learning algorithm.** [[2]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502494&forceview=1#footnote-2)

Because Q-learning is an off-policy method that learns about the optimal policy directly, we no longer use the value of the action 𝐴′ that our agent's policy would choose in state 𝑆′ as part of our update rule, and instead use the value of the best action available in 𝑆′. This means that, unlike with Sarsa, we do not carry the next-action used as part of the update forward and execute it in the next iteration.

Q-learning is guaranteed to converge to the optimal policy as long as each state-action pair is experienced an infinite number of times, and with a learning rate 𝛼 which decreases to 0 over time.Sarsa is also guaranteed to converge to the optimal policy under these conditions, as long as the agent's policy becomes greedy over time, otherwise, it will converge to a near-optimal policy which explores.

# Lesson 2 Summary

Completion requirements

In this lesson, we have introduced two fundamental temporal-difference learning control methods. We have discussed

- how temporal-difference learning can be used to learn action-values;
- how to implement Sarsa, an on-policy TD control algorithm; and
- how to implement Q-Learning, an off-policy TD control algorithm.

![](https://moodle.bath.ac.uk/theme/image.php/boost_union/page/1793686586/monologo?filtericon=1)

# Lesson 3 Introduction

Completion requirements

We're going to end this week by comparing the performance of the **Sarsa** and **Q-learning** algorithms on the cliff walking problem. This should help us better understand the potential advantages and disadvantages of on-policy and off-policy algorithms.

# The Cliff Walking Problem: Policies Learned

Completion requirements

Recall the **cliff walking** problem, illustrated below, that we looked at previously when first introducing on-policy and off-policy methods.

This environment is a gridworld, and the agent can choose to go North, South, East, and West in each state. All actions in this environment are deterministic, and take the agent in its intended direction with probability 1. Moving into a wall causes the agent to remain in its current grid square, and moving into 'The Cliff' causes the agent to be moved back to the starting grid square. In this problem, the agent earns a reward of −1 for each action taken, except for actions that cause it to move into "The Cliff", which result in a reward of −100.

**The cliff walking Problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502499&forceview=1#footnote-1)

The route taken by the optimal policy in this environment is shown in red and involves taking the shortest path along the cliff edge to the goal. This is the route that a Q-Learning agent trained using an 𝜖-greedy behavioural policy (𝜖=0.1) would learn to take. If we trained this same agent using Sarsa, it would learn a policy that takes the safer route to the goal state, marked in blue. Why do these two algorithms learn to take these two different routes? The answer lies in their different update rules.

Because it is behaving according to an 𝜖-greedy policy, our agent has a chance to choose an exploratory action that causes it to fall off the cliff and earn a large negative reward whenever it is near the cliff edge.

Sarsa is an on-policy algorithm, so it is limited to learning about the policy it is following (which, in this case, is a policy that explores randomly with probability 0.1). So, Sarsa takes these exploratory actions that cause our agent to fall off the cliff into account when learning. Therefore, the near-optimal policy learned by Sarsa tries to avoid accidentally falling off the cliff due to exploration by staying away from the cliff edge.

Consider the action-value function 𝑄(𝑠,𝑎) that Sarsa is learning.  
_"What is the return I can expect to earn if I take action 𝑎 in state 𝑠, then follow **an 𝜖-greedy policy**thereafter?"_

On the other hand, Q-learning learns about the optimal policy, which always acts greedily and never explores, even though its behaviour policy does explore and occasionally falls off the cliff. This is because the Q-learning update rule only considers the highest-valued action available in the next-state, instead of the (potentially exploratory) action that the behaviour policy would take.

Consider the action-value function 𝑄(𝑠,𝑎) that Q-Learning is learning.  
_"What is the return I can expect to earn if I take action 𝑎 in state 𝑠, then follow **the optimal policy**thereafter?"_

# The Cliff Walking Problem: Learning Performance

Completion requirements

We have now seen the policies that Sarsa and Q-learning learn when applied to the cliff walking problem for an agent following an 𝜖-greedy policy. We are now going to take a look at the returns earned by these agents during each training episode. These results are shown in the graph below.

**Learning curves showing the rewards earned each episode during training by Sarsa and Q-learning agents on the cliff walking problem.** [[1]](https://moodle.bath.ac.uk/mod/page/view.php?id=1502500&forceview=1#footnote-1)

You might initially be surprised to see that Q-learning performs significantly worse than Sarsa during training, despite learning a more optimal path. However, once you consider that this graph shows the rewards earned _during training_, this result makes perfect sense.

Recall that both the agents generate experience using an 𝜖-greedy policy that selects actions at random with probability 0.1. Even though our Q-Learning agent does not take exploratory actions into account in its updates, they still occur during training. In fact, because the Q-learning agent learns the optimal route, it spends more time closer to the cliff edge during training than the Sarsa agent. This causes the Q-learning agent to fall off the cliff more often and earn a lower reward on average than the Sarsa agent during training.

Because the Sarsa agent takes random exploration into account as part of the safer policy that it learns, it spends less time close to the cliff edge than the Q-Learning agent. This means it falls off the cliff less often and earns a higher return during training, even though it takes a longer route to reach the goal.

This is the cause of the strange situation where, during training, Q-learning seems to perform worse than Sarsa, even though it learns the optimal policy. If we were to disable exploration entirely and act greedily with respect to the action-value functions learned by the two algorithms, we would likely observe that Q-Learning performs better, as we would expect.

# Lesson 3 Summary

Completion requirements

In this lesson, we have examined the performance of Q-Learning and Sarsa algorithms when applied to the cliff walking problem. We discussed

- how, in some environments, exploration in agent's policy can cause on-policy and off-policy methods to learn significantly different final policies;
- how, despite being able to learn optimal policies directly, off-policy agents can display reduced performance during training compared to on-policy methods; and
- how to interpret learning curves when comparing the performance on on-policy and off-policy methods during training.
# Weekly Summary

Completion requirements

This week, we introduced temporal-difference learning methods as an alternative to dynamic programming and Monte Carlo methods. Like Monte Carlo methods, they learn using experience generated by our agent without relying on a model of our agent's environment. However, like dynamic programming methods, they used bootstrapping, which allows them to learn from fragments of experience as short as a single time step. Together, this allows temporal-difference methods to learn from our agent's experiences without being limited to learning from full sample returns.

We also took a close look at the differences between the updates performed by Monte Carlo and temporal-difference methods. We discovered that temporal difference methods consistently outperform Monte Carlo methods in practice, even on simple problems, because they can learn estimates that take into account the sequentiality of the problem the agent is solving.

This week, we introduced three concrete temporal difference methods. We started by examining TD(0), a simple temporal-difference method for policy evaluation, before moving on to two control algorithms, Sarsa and Q-Learning. Sarsa is an on-policy control algorithm, whereas Q-Learning is an off-policy control algorithm.

We finished by looking at how the Sarsa and Q-Learning algorithms performed when applied to the cliff walking problem. This gave us a good opportunity to examine the differences between on-policy and off-policy methods in general. We observed that Sarsa converged upon a more conservative policy because it learned to avoid negative rewards incurred by the random exploration inherent to its behavioural policy. On the other hand, we saw that Q-Learning learned an optimal policy, but incurred more negative rewards during training.

Next week, we will consider two extensions to Sarsa and Q-Learning aimed at improving their sample efficiency, allowing them to learn more from a given amount of experience.