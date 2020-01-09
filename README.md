# UNN.js
Deep Learning in Javascript. Alternative to ConvNetJS, that is 4x faster.

UNN lets you generate, train and test artifficial neural networks. It can also load existing models from ConvNetJS (.json) and Caffe (.caffemodel). It consists of two files:

* `UNN` - main tools for training and testing
* `UNN.util` - UNN utilities, contains parsers for other formats

## Documentation
Each layer of the network holds data in a cuboid: 3D matrix W x H x D. We can think of it as D feature maps of size W x H. Values are stored in a linear array, arranged by Z, then by Y, then by X.

Each layer has a type: an array of parameters of the layer

    [LTYP, FUNC, W, H, D]
    
- `LTYP`: layer type: `"inpt"`, `"conv"`, `"pool"`, `"full"`
- `FUNC`: activation function: `"line"`, `"sigm"`, `"tanh"`, `"relu"`, `"sfmx"`
- `W`, `H`, `D`: width, height and depth of the data in a layer

In addition, `"conv"` and `"pool"` layers have three extra parameters:

    [...  K, S, P]

- `K`, `S`, `P`: kernel size, stride and padding

#### `UNN.Create(Types, V)`
- `Types` - Array of types of layers in the network
- `V` - the network will be filled with random values between `-V` and `V`
- returns a neural network

A network for detecting XOR, architecture 2:2:1, using Sigmoid for activation:

    var net = UNN.Create([ ["inpt","line",2,1,1],["full","sigm",2,1,1],["full","sigm",1,1,1] ], 0.5);
    
#### `UNN.Train(net, inputs, outputs, prm)`
- `net` - a network made by UNN.Create(), or achieved otherwise
- `inputs` - an array of vectors for the input
- `outputs` - an array of vectors expected at the output
- `prm` - training parameters, `{ method:"sgd"|"momentum"|"adagrad"|"adadelta", batch_size:Number }`
- returns an error of the network after training

Let's train our network on four possible inputs

    var In = [[0,0],[0,1],[1,0],[1,1]], Ou = [[0],[1],[1],[0]];
    var prm = { method:"sgd", batch_size:1 };
    for(var i=0; i<5000; i++) UNN.Train(net,In,Ou,prm);
    
#### `UNN.GetOutput(net, vec, O)`
- `net` - a network
- `vec` - an input vector
- `O` - array - a container for outputs
- fills O with an output of each layer, you can reuse the same O between multiple calls (to be GC-friendly)

Let's test our network for XOR

    var O = [];
	UNN.GetOutput(net, [0,0], O);  console.log(O[2][0]);  // prints [0.016312343710744647]
	UNN.GetOutput(net, [0,1], O);  console.log(O[2][0]);  // prints [0.9824375045125838]
	
#### Saving and Loading

UNN networks are simple objects, which can be saved to a string with JSON.stringify(), and parsed with JSON.parse().

## Testing on MNIST
Let's make the same network for MNIST, as in [ConvNetJS demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html), which has 5994 weights.

    var Types = [
        ["inpt","line",24,24, 1],
        ["conv","relu",24,24, 8, 5,1,2], ["pool","line",12,12, 8, 2,2], 
        ["conv","relu",12,12,16, 5,1,2], ["pool","line", 4, 4,16, 3,3],
        ["full","sfmx",10, 1, 1]      ];
    var prm = {method:"adadelta", batch_size:20};
    
One training loop over 60 000 values takes 61 seconds and makes 143 mistakes when testing (test error rate 1.43%).
The same training loop with ConvNetJS takes 246 seconds and makes 145 mistakes when testing (test error rate 1.45%). After four such iterations, both networks make about 110 mistakes.
 
We made the same test with TensorFlow.js (which uses GPU through WebGL). One iteration took only 52 seconds (on our GPU). But the network made 220 errors after one iteration, and 150 errors after four iterations (208 seconds).

## Loading other formats

#### `UNN.util.fromCNJS(obj)`
Takes an object: a ConvNetJS network (e.g. parsed from a JSON file). Converts it into UNN network.

#### `UNN.util.fromCaffe(arrayBuffer)`
Takes an ArrayBuffer of a .caffemodel file. Converts it into UNN network.

We used it to parse and test a [LeNet model](https://github.com/mravendi/caffe-test-mnist-jpg/blob/master/model/lenet_iter_10000.caffemodel) (431080 weights), which made only 94 mistakes on MNIST.
