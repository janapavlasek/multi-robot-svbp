import os
import numpy as np
import matplotlib.pyplot as plt


class PointSwarm(object):
    """A swarm consisting of linear 2D point robots."""

    def __init__(self, n_agents, dt=0.01, dim=2, comm_radius=1, collison_radius=0.5, ctrl_space='acc',
                 lims=(-3, 3), cmd_lim=None, vel_lim=None, init_vel_range=(0.1, 1), start_state=None,
                 out_path="output"):
        assert ctrl_space in ['acc', 'vel'], "Control space must be one of ['acc', 'vel']"

        self.n_agents = n_agents
        self.dim = dim
        self.dt = dt
        self.comm_radius = comm_radius
        self.collison_radius = collison_radius
        self.ctrl_space = ctrl_space
        self.x_dim = dim * 2 if ctrl_space == 'acc' else dim
        self.lims = lims
        self.cmd_lim = cmd_lim
        self.vel_lim = vel_lim
        self.out_path = out_path
        self.iter = 0

        self.vel_range = init_vel_range  # m / s

        if start_state is None:
            self.state, self.graph = self.init_state()
        else:
            self.state = start_state.copy()
            # Assert size is correct.
            assert self.state.shape[0] == self.n_agents, "Must provide same number of start states as agents." \
                                                         f"{self.state.shape[0]} != {self.n_agents}"
            assert self.state.shape[1] == self.x_dim, "Start state has incorrect dimension, " \
                                                      f"{self.state.shape[1]} != {self.x_dim}"
            self.graph = self.calculate_adjacency(self.state[:, :self.dim])

        # Keep a copy of the start state for resetting.
        self.start_state = self.state.copy()

        # Model.
        self._A = np.eye(self.x_dim)
        self._B = self.dt * np.eye(self.dim)
        if ctrl_space == 'acc':
            self._A[:dim, dim:] = self.dt * np.eye(self.dim)
            self._B = np.concatenate((0.5 * self.dt**2 * np.eye(self.dim), self._B), axis=0)
        self._A = self._A.T
        self._B = self._B.T

        # Drawing helpers.
        self.fig = None
        self.ax = None
        self.plt_pts = None
        self.vel_arrows = None

    def reset(self):
        self.iter = 0
        self.state = self.start_state

    def get_state(self):
        return self.state, self.graph

    def init_state(self, full_conn=False):
        state = np.zeros((self.n_agents, self.x_dim))
        # Initialize positions.
        state[:, :self.dim] = np.random.uniform(*self.lims, size=(self.n_agents, self.dim))
        if self.ctrl_space == 'acc':
            # Initialize velocities.
            state[:, self.dim:] = np.random.uniform(*self.vel_range, size=(self.n_agents, self.dim))
            # Make some of the velocities negative.
            state[:, self.dim:] *= np.random.choice([-1, 1], size=(self.n_agents, self.dim))

        graph = self.calculate_adjacency(state[:, :self.dim])

        if full_conn:
            # Ensure the graph is fully connected.
            lost_robots = self.lost_robots()
            collisons, _ = np.nonzero(self.calculate_collisions(state[:, :self.dim]))
            reset = np.unique(np.concatenate([lost_robots, collisons]))
            while len(reset) > 0:
                state[reset, :self.dim] = np.random.uniform(*self.lims, size=(len(reset), self.dim))
                graph = self.calculate_adjacency(state[:, :self.dim])
                lost_robots = self.lost_robots()
                collisons, _ = np.nonzero(self.calculate_collisions(state[:, :self.dim]))
                reset = np.unique(np.concatenate([lost_robots, collisons]))

        return state, graph

    def step(self, action):
        """Stepping function.

        Args:
            action: Acceleration in x and y, for each robot (N, 2)
        """
        if self.cmd_lim is not None:
            # Limit the commands.
            cmd_norm = np.sqrt((action * action).sum(-1))
            if (cmd_norm > self.cmd_lim).any():
                scale = self.cmd_lim / cmd_norm[cmd_norm > self.cmd_lim]
                action[cmd_norm > self.cmd_lim] *= np.expand_dims(scale, axis=1)

        # Update the state.
        self.state = self.state @ self._A + action @ self._B

        if self.vel_lim is not None and self.ctrl_space == 'acc':
            vel_norm = np.sqrt((self.state[:, self.dim:] * self.state[:, self.dim:]).sum(-1))
            if (vel_norm > self.vel_lim).any():
                scale = self.vel_lim / vel_norm[vel_norm > self.vel_lim]
                self.state[vel_norm > self.vel_lim, self.dim:] *= np.expand_dims(scale, axis=1)

        # Calculate the adjacency graph.
        self.graph = self.calculate_adjacency(self.state[:, :self.dim])

        self.iter += 1

        return self.state, self.graph

    def render(self, vels=True, edges=False, save=False):
        if self.fig is None:
            # Setup drawing.
            with plt.ion():
                self.fig = plt.figure(0)
                self.ax = self.fig.add_subplot(xlim=self.lims, ylim=self.lims, aspect='equal')
                self.fig.tight_layout()
                self.plt_pts, = self.ax.plot(self.state[:, 0], self.state[:, 1], 'bo', zorder=3)

                if vels:
                    xs, ys, dxs, dys = self.calc_vel_arrows(self.state)
                    self.vel_arrows = [self.ax.arrow(xs[i], ys[i], dxs[i], dys[i], color='r', head_width=0.1, zorder=2)
                                       for i in range(self.n_agents)]
        else:
            # Remove edges, if any.
            while len(self.ax.lines) > 1:
                line = self.ax.lines[-1]
                line.remove()
            # Update the robots.
            self.plt_pts.set_xdata(self.state[:, 0])
            self.plt_pts.set_ydata(self.state[:, 1])

            if vels:
                # Draw new arrows.
                xs, ys, dxs, dys = self.calc_vel_arrows(self.state)
                for i in range(self.n_agents):
                    self.vel_arrows[i].set_data(x=xs[i], y=ys[i], dx=dxs[i], dy=dys[i])

        if edges:
            # Draw the edges.
            xs, ys = self.calc_edge_segments(self.graph)
            self.ax.plot(xs, ys, 'g', zorder=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if save:
            out_file = os.path.join(os.path.expanduser(self.out_path), f'env_{self.iter:04}.png')
            self.fig.savefig(out_file, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)

    def calculate_distances(self, pos):
        delta = pos[...,None,:] - pos
        dists = np.linalg.norm(delta, axis=-1)
        return dists

    def calculate_adjacency(self, pos):
        dists = self.calculate_distances(pos)
        adj = dists <= self.comm_radius
        np.fill_diagonal(adj, 0)
        return adj

    def calculate_collisions(self, pos):
        dists = self.calculate_distances(pos)
        collisions = dists <= self.collison_radius
        np.fill_diagonal(collisions, 0)
        return collisions

    def calc_edge_segments(self, graph):
        # Zero out duplicate connections and self connections.
        graph = np.triu(graph)
        np.fill_diagonal(graph, 0)
        # Get all the pairs of nodes making an edge.
        n1, n2 = graph.nonzero()
        xs = np.stack([self.state[n1, 0], self.state[n2, 0]], axis=0)
        ys = np.stack([self.state[n1, 1], self.state[n2, 1]], axis=0)
        return xs, ys

    def calc_vel_arrows(self, states, mag=0.4):
        x, y = states[:, 0], states[:, 1]
        vx, vy = states[:, 2], states[:, 3]
        norm_v = np.sqrt(vx * vx + vy * vy)
        # Catch divide by zero.
        norm_v[np.isclose(norm_v, 0)] = 1
        # Normalize the vector.
        vx, vy = mag * vx / norm_v, mag * vy / norm_v
        return x, y, vx, vy

    def lost_robots(self):
        conn = self.graph.sum(axis=1)
        return np.nonzero(conn < 1)[0]
