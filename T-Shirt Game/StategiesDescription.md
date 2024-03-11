- *Simple Majority Strategy*

At first I implemented the system to have pairs and interchange color information and based on the last t-shirt seen decide which t-shirt to wear, but I didn´t get interesting results so I changed it to the final presented version where an agent looks around it and based on the majority decides which t-shirt to wear in the next round.

This strategy has a straightforward decision-making process where agents choose their state based on the most common state observed in their immediate environment. Each player observes the T-shirt colors of nearby players within a certain radius and then adopts the color worn by the majority. This strategy sucessfully demostrated the a social norm generation and reflect global patterns or consensus without centralized control. Each round of observations and decisions gradually led the entire group to converge on a single color. Showing how simple rules can lead to complex behaviors.

An important aspect of this strategy is the observation range, which determines how many other players an agent considers when deciding its next action. Adjusting the observation range can significantly affect the speed and likelihood of achieving convergence. A too-small range (tested with a value of 3) might limit the information available for making decisions, slowing down the process of reaching a consensus. On the other hand, too large an area of distribution could make local variations in color distribution less influential, possibly leading to a deadlock if the population is equally divided. 

Using the value of five observable players in each round after many iterations, convergence was reached at iteration 55 on average.

- *History-Based Strategy*

The History-Based Strategy was a bit more complex to implement and includes the use of memory to the decision-making process by allowing players to base their choices on the entire history of observations, rather than just the most recent round. In this approach, each player keeps a record of all the T-shirt colors they have observed in other players throughout the game. When deciding which color to wear next, a player analyzes this cumulative history and chooses the color they have seen the most. This strategy shows a form of learning and adaptation, as players' decisions are influenced by its previous experiences.

