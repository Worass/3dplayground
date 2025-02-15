// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs';

class RLAgent {
    constructor() {
        this.model = this.createModel();
        this.optimizer = tf.train.adam(0.001); // Adjusted learning rate
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [4] })); // Increased units
        model.add(tf.layers.dense({ units: 64, activation: 'relu' })); // Increased units
        model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
        model.compile({ loss: 'meanSquaredError', optimizer: this.optimizer });
        return model;
    }

    async predict(state) {
        return tf.tidy(() => {
            const input = tf.tensor2d([state]);
            return this.model.predict(input).dataSync();
        });
    }

    async train(states, actions, rewards, nextStates) {
        const xs = tf.tensor2d(states);
        const ys = tf.tensor2d(actions.map((action, index) => {
            const reward = rewards[index];
            const nextState = nextStates[index];
            const nextActionProbs = this.predict(nextState);
            const maxNextActionProb = Math.max(...nextActionProbs);
            return action.map((a, i) => i === action.indexOf(1) ? reward + 0.99 * maxNextActionProb : a);
        }));
        await this.model.fit(xs, ys, { epochs: 20 }); // Increased epochs
    }
}

class BlobEnvironment {
    constructor() {
        this.reset();
    }

    reset() {
        this.blob = { x: 0, y: 0 };
        this.goal = { x: Math.random() * 10, y: Math.random() * 10 };
        return [this.blob.x, this.blob.y, this.goal.x, this.goal.y];
    }

    step(action) {
        if (action === 0) this.blob.y += 1;
        else if (action === 1) this.blob.y -= 1;
        else if (action === 2) this.blob.x += 1;
        else if (action === 3) this.blob.x -= 1;

        const distance = Math.hypot(this.goal.x - this.blob.x, this.goal.y - this.blob.y);
        const reward = -distance;

        if (distance < 1) {
            return [[this.blob.x, this.blob.y, this.goal.x, this.goal.y], 10]; // Positive reward for reaching the goal
        }

        return [[this.blob.x, this.blob.y, this.goal.x, this.goal.y], reward];
    }
}

const agent = new RLAgent();
const env = new BlobEnvironment();

async function trainAgent() {
    const experiences = [];
    for (let i = 0; i < 1000; i++) {
        let state = env.reset();
        let totalReward = 0;
        for (let j = 0; j < 100; j++) {
            const actionProbs = await agent.predict(state);
            const action = actionProbs.indexOf(Math.max(...actionProbs));
            const [nextState, reward] = env.step(action);
            totalReward += reward;
            experiences.push({ state, action, reward, nextState });
            state = nextState;
            if (reward === 10) break; // Goal reached
        }
        if (experiences.length > 64) { // Increased batch size
            const batch = experiences.splice(0, 64);
            await agent.train(
                batch.map(e => e.state),
                batch.map(e => [0, 0, 0, 0].map((_, i) => i === e.action ? 1 : 0)),
                batch.map(e => e.reward),
                batch.map(e => e.nextState)
            );
        }
        console.log(`Episode ${i} - Total Reward: ${totalReward}`);
    }
}

trainAgent();
