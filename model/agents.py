'''
Mesa epstein civil violence model agents with added network dynamics. 
Additional modifications were made so that the base model is closer to the Netlogo version, which produces results more similar to the original paper.
'''

import math
from enum import Enum
import mesa


class CitizenState(Enum):
    ACTIVE = 1
    QUIET = 2
    ARRESTED = 3


# update_neighbors() and move() were both modified to fall in line with the 
class EpsteinAgent(mesa.discrete_space.CellAgent):
    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.cell.get_neighborhood(radius=self.vision)
        self.neighbors = self.neighborhood.agents

    def move(self):
        
        self.neighborhood = self.cell.get_neighborhood(radius=self.vision)
        self.neighbors = self.neighborhood.agents
        
        self.empty_neighbors = []
        
        # empty cells and cells with only arrested citizens are considered candidates to move to
        for c in self.neighborhood:
            if c.is_empty:
                self.empty_neighbors.append(c)
            else:
                if all(getattr(agent, "state", None) == CitizenState.ARRESTED for agent in c.agents):
                    self.empty_neighbors.append(c)
                    
        if self.model.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.move_to(new_pos)


class Citizen(EpsteinAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self, model, regime_legitimacy, threshold, vision, arrest_prob_constant
    ):
        """
        Create a new Citizen.
        Args:
            model: the model to which the agent belongs
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(model)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.threshold = threshold
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None

        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []
        
    # NETWORK --------
    def set_network_neighbors(self, node):
        
        self.network_neighbors = self.model.citizen_network.get_neighbors(node)
    # ----------------
        
    def step(self):
        """
        Decide whether to activate.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.
        self.update_neighbors()
        self.update_estimated_arrest_probability()

        net_risk = self.risk_aversion * self.arrest_probability
        if (self.grievance - net_risk) > self.threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself
        for neighbor in self.neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1
        
        # NETWORK ------------------
        if self.model.networked:
            for neighbor in self.network_neighbors:
                if isinstance(neighbor, Citizen):
                    if neighbor.state == CitizenState.ACTIVE:
                        actives_in_vision += 1
        # ---------------------------
        
        
        # changed to floor instead of round
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * (cops_in_vision // actives_in_vision)
        )


class Cop(EpsteinAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision, arrest a random active agent and move to their position.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(self, model, vision, max_jail_term):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(model)
        self.vision = vision
        self.max_jail_term = max_jail_term

    def step(self):
        """
        Inspect local vision, arrest a random active agent and move to their position.
        """
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE:
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            arrestee.jail_sentence = self.random.randint(0, self.max_jail_term)
            arrestee.state = CitizenState.ARRESTED
            self.move_to(arrestee.cell)
