'''
Mesa epstein civil violence model with added network dynamics. 
Additional modifications were made so that the base model is closer to the Netlogo version, which produces results more similar to the original paper.
'''


import mesa
import networkx as nx
from model.agents import (
    Citizen,
    CitizenState,
    Cop,
)


class EpsteinCivilViolence(mesa.Model):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full

    Args:
        height: grid height
        width: grid width
        citizen_density: approximate % of cells occupied by citizens.
        cop_density: approximate % of cells occupied by cops.
        citizen_vision: number of cells in each direction (N, S, E and W) that
            citizen can inspect
        cop_vision: number of cells in each direction (N, S, E and W) that cop
            can inspect
        legitimacy:  (L) citizens' perception of regime legitimacy, equal
            across all citizens
        max_jail_term: (J_max)
        active_threshold: if (grievance - (risk_aversion * arrest_probability))
            > threshold, citizen rebels
        arrest_prob_constant: set to ensure agents make plausible arrest
            probability estimates
        movement: binary, whether agents try to move at step end
        max_iters: model may not have a natural stopping point, so we set a
            max.
        networked: whether the social network is added or not
    """

    def __init__(
        self,
        width=40,
        height=40,
        citizen_density=0.7,
        cop_density=0.074,
        citizen_vision=7,
        cop_vision=7,
        legitimacy=0.8,
        max_jail_term=1000,
        active_threshold=0.1,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=1000,
        seed=None,
        networked=True,
        m=10
    ):
        super().__init__(seed=seed)
        self.movement = movement
        self.max_iters = max_iters
        self.networked = networked

        self.grid = mesa.discrete_space.OrthogonalVonNeumannGrid(
            (width, height), capacity=20, torus=True, random=self.random
        )

        model_reporters = {
            "active": CitizenState.ACTIVE.name,
            "quiet": CitizenState.QUIET.name,
            "arrested": CitizenState.ARRESTED.name,
            "tension": "TENSION"
        }
        
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters
        )
        if cop_density + citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")

        for cell in self.grid.all_cells:
            klass = self.random.choices(
                [Citizen, Cop, None],
                cum_weights=[citizen_density, citizen_density + cop_density, 1],
            )[0]

            if klass == Cop:
                cop = Cop(self, vision=cop_vision, max_jail_term=max_jail_term)
                cop.move_to(cell)
            elif klass == Citizen:
                citizen = Citizen(
                    self,
                    regime_legitimacy=legitimacy,
                    threshold=active_threshold,
                    vision=citizen_vision,
                    arrest_prob_constant=arrest_prob_constant,
                )
                citizen.move_to(cell)

        
        # region NETWORK -----------------------
        if networked:
            citizens = [agent for agent in self.agents if isinstance(agent, Citizen)]
            num_citizens = len(citizens)

            nx_graph = nx.barabasi_albert_graph(num_citizens, m, seed=seed) # generation parameter is fixed at 10

            self.citizen_network = mesa.space.NetworkGrid(g=nx_graph)
        
            for citizen, node in zip(citizens, self.citizen_network.G.nodes()):
                self.citizen_network.place_agent(citizen, node)
                
            for citizen, node in zip(citizens, self.citizen_network.G.nodes()):
                citizen.set_network_neighbors(node)
        # -------------------------------
        
        self.running = True
        self._update_counts()
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("step")
        self._update_counts()
        self.datacollector.collect(self)

        if self.steps > self.max_iters:
            self.running = False

    def _update_counts(self):
        """Helper function for counting nr. of citizens in given state, as well as calculate the tension at given step."""
        counts = self.agents_by_type[Citizen].groupby("state").count()

        for state in CitizenState:
            setattr(self, state.name, counts.get(state, 0))
            
        citizens= self.agents_by_type[Citizen]
        n=len(citizens)
        if n:                                     
            avg_G= sum(a.grievance for a in citizens)/n
            avg_R=sum(a.risk_aversion for a in citizens)/n
            prop_quiet= counts.get(CitizenState.QUIET, 0)/n
            if avg_R != 0:
                self.TENSION = (avg_G * prop_quiet / avg_R)
            else:
                self.TENSION = 0

        else:
            self.TENSION = 0
