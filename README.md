# UNN.js
Deep Learning in Javascript. Alternative to ConvNetJS, that is 5x faster.

UNN lets you generate, train and test artifficial neural networks. It can also load existing models from ConvNetJS (.json) and Caffe (.caffemodel). It consists of two files:

* `UNN` - main tools for training and testing
* `UNN.util` - UNN utilities, contains parsers for other formats

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

    var Types = [  
        ["inpt","line",2,1,1], 
        ["full","sigm",2,1,1], 
        ["full","sigm",1,1,1]  
    ];
    var net = UNN.Create(Types, 0.1);
    
