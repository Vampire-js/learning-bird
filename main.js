const canvas = document.getElementById("canvas");
const c = canvas.getContext("2d");
canvas.width = innerWidth;
canvas.height = innerHeight;

let score = 0
let best = 0

class Pipe {
  constructor() {
    this.w = 100;
    this.h = 600;
    this.x = canvas.width/2 - 100;
    this.y = 300 + 200 * Math.random();
  }
  draw() {
    c.fillStyle = "green";
    c.fillRect(this.x, this.y, this.w, this.h);
  }
  update() {
    this.x -= 3;
    this.draw();
  }
}

let pipes = [];
let topEdge = new Pipe()
topEdge.w = canvas.width
topEdge.h = 10
topEdge.x = 0
topEdge.y = 10

pipes.push(topEdge)
let pipesInterval;

function startPipeSpawning() {
  pipesInterval = setInterval(() => {
    pipes.push(new Pipe());
    score++
  }, 1000);
}

class Player {
  constructor() {
    this.w = 50;
    this.h = 50;
    this.position = { x: 80, y: canvas.height / 2 };
    this.velocity = { x: 0, y: 0 };
  }
  draw() {
    c.fillStyle = "yellow";
    c.fillRect(this.position.x, this.position.y, this.w, this.h);
  }
  move() {
    this.velocity.y = -1;
  }
  reset(){
    this.position = { x: 80, y: canvas.height / 2 };

    this.velocity = {x:0, y:0}
  }
  update() {
    this.position.y += this.velocity.y;
    this.velocity.y += 0.01; 
    this.draw();
  }
}

let player = new Player();

function resetGame() {
  score = 0
  player.reset()
  pipes = [];
  clearInterval(pipesInterval);
  startPipeSpawning();
 
}

function GameOver() {
  for (let e of pipes) {
    if (
      player.position.x + player.w >= e.x &&
      player.position.x <= e.x + e.w &&
      player.position.y + player.h >= e.y
    ) {
      return true;
    }
  }
  if (player.position.y + player.h > canvas.height || player.position.y < 0) {
    return true;
  }
  return false;
}

function GameState() {
  let nextPipe = pipes.find(p => p.x < player.position.x + player.w);
  if (!nextPipe) {
    nextPipe = new Pipe();
    nextPipe.x = canvas.width;
  }
  const pipeCenterY = nextPipe.y - 100;
  const playerCenterY = player.position.y + player.h / 2;
  const dy = (playerCenterY - pipeCenterY) / canvas.height;
  
  return [
    (playerCenterY )/ canvas.height,
    player.velocity.y,
    (nextPipe.x - player.position.x) / canvas.width,
    dy
  ];
}

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [4], units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 16, activation: 'sigmoid' }));
model.add(tf.layers.dense({ units: 2 }));
model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

async function chooseAction(state, epsilon = 0.5) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * 2);
  const prediction = model.predict(tf.tensor2d([state]));
  const data = await prediction.data();
  prediction.dispose();
  return data[0] > data[1] ? 0 : 1;
}

async function train(state, action, reward, nextState, done, gamma = 0.9) {
  const stateTensor = tf.tensor2d([state]);
  const nextStateTensor = tf.tensor2d([nextState]);
  const qValues = model.predict(stateTensor);
  const nextQ = model.predict(nextStateTensor);
  const qData = await qValues.data();
  const nextQData = await nextQ.data();
  let target = [...qData];
  target[action] = reward /*+ (done ? 0 : gamma * Math.max(...nextQData))*/;
  const targetTensor = tf.tensor2d([target]);
  await model.fit(stateTensor, targetTensor, { epochs: 1, verbose: 0 });
  tf.dispose([stateTensor, nextStateTensor, qValues, nextQ, targetTensor]);
}

let epsilon = .4;
const epsilonDecay = 0.995;
const epsilonMin = 0.1;
let episode = 0;
document.getElementById("iter").innerHTML = `Iteration: ${episode}`
document.getElementById("score").innerHTML = `Score: ${score}`


async function logicLoop() {
  while (true) {
    const state = GameState();
    const action = await chooseAction(state, epsilon);
document.getElementById("score").innerHTML = `Score: ${score}`
best = Math.max(best, score)
document.getElementById("best").innerHTML = `Best Score: ${best}`

    
    if (action === 1) player.move();

    const reward = GameOver() ? 0 : 1;
    const nextState = GameState();
    const done = GameOver();

    await train(state, action, reward, nextState, done);

    if (epsilon > epsilonMin) epsilon *= epsilonDecay;

    if (done) {
      episode++
      document.getElementById("iter").innerHTML = `Iteration: ${episode}`
      resetGame();
    }
 }
}



function animate() {
  
  c.fillStyle = "skyblue";
  c.fillRect(0, 0, canvas.width, canvas.height);

  player.update();
  pipes = pipes.filter(p => p.x + p.w > 0);
  pipes.forEach(p => p.update());

  requestAnimationFrame(animate);
}
resetGame()

animate();    
logicLoop();  

