const mnist = require('mnist')(false)
const syft = require('syft')

let testSamples = 10000
let trainingSamples = 60000

let dataset = mnist(trainingSamples, testSamples)

async function train() {
  let training = {
    input: await syft.Tensor.FloatTensor.create(dataset.training.input),
    output: await syft.Tensor.FloatTensor.create(dataset.training.output)
  }

  let testing = {
    input: await syft.Tensor.FloatTensor.create(dataset.test.input),
    output: await syft.Tensor.FloatTensor.create(dataset.test.output)
  }

  let model = await syft.Model.Sequential.create([
    await syft.Model.Linear.create(784, 10)
  ])

  let criterion = await syft.Model.CrossEntropyLoss.create()
  let optim = await syft.Optimizer.SGD.create(await model.parameters(), 0.06)
  let metric = ['accuracy']
  let softmax = await syft.Model.Softmax.create()

  let loss = await model.fit(
    training.input,
    training.output,
    criterion,
    optim,
    32,     // batch size
    2,      // interactions
    1,      // log interval
    metric,
    true    // verbose
  )

  console.log('Trained with a final loss:', loss)

  let perd = await softmax.forward(
    await model.forward(testing.input)
  )

  // select a random test example to draw
  let select = Math.floor(testSamples * Math.random())

  dataset.draw(
    (await testing.input.to_numpy()).slice(select * 784, (select + 1) * 784),
    (await perd.to_numpy()).slice(select * 10, (select + 1) * 10),
    (await testing.output.to_numpy()).slice(select * 10, (select + 1) * 10)
  )
}

train()
  .then(() => console.log('Done'))
  .catch((err) => console.log(err))
