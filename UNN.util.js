	
	UNN.util={};
	
	UNN.util.weightCount = function(net) {
		var wcnt=0;
		for(var j=0; j<net.length; j++) {  var tc=0;
			for(var k=0; k<net[j].ws.length; k++) tc+=net[j].ws[k].length;
			wcnt +=tc;  //console.log(net[j],tc);
		}
		return wcnt;
	}


	UNN.util.fromCNJS = function(net) {
		function getTyp(lay, lbl,fnc) {
			var w=lay["out_sx"], h=lay["out_sy"], d=lay["out_depth"];
			if(lbl=="full") {  w=Math.max(w,h,d);  h=d=1;  }
			var out = [lbl,fnc,w, h, d];
			if(lbl!="full"&&lbl!="inpt") out.push(lay["sx"],lay["stride"]);
			if(lbl=="conv") out.push(lay["pad"]);
			return out;
		}
		
		net = net.layers;
		var ptyp = getTyp(net[0],"inpt","line");
		var nnet = [{typ:ptyp, ws:[]}];
		for(var i=1; i<net.length; i++) {
			var lay = net[i], type=lay.layer_type, tt="line";
			if(type=="fc") type="full";
			if(type!="pool") {
				i++;
				var ot = net[i].layer_type, tt = {"softmax":"sfmx","sigmoid":"sigm","relu":"relu"}[ot];
				if(tt==null) throw ot;
			}
				
			var typ = getTyp(lay,type,tt), ws=[];  nnet.push({typ:typ, ws:ws});
			
			var pt = nnet[nnet.length-2].typ, pw=pt[2],ph=pt[3],pd=pt[4];  if(type=="conv") pw=ph=typ[5];
			
			if(type!="pool")
				for(var j=0; j<lay.filters.length; j++) {
					var fj = lay.filters[j].w, nws=new UNN.Array(pd*pw*ph+1);  ws.push(nws); 
					var vls = [];
					for(var pp in fj) {  vls.push(fj[pp]);  }
					
					for(var z=0;z<pd;z++)
						for(var y=0;y<ph;y++)
							for(var x=0;x<pw;x++) nws[z*pw*ph + y*pw + x] = vls[pd*(y*pw + x) + z];
					
					nws[nws.length-1] = lay.biases.w[""+j];
				}
		}
		//console.log(nnet);
		return nnet;
	}
	UNN.util.fromCaffe = function(buf) {
		var lrs = UNN.util.parseProtobuf(new Uint8Array(buf),0,"NetParameter").layer;
		
		var w=0, h=0, d=0, dlen=0;  // the next loop only finds the input dimension
		for(var li=lrs.length-1; li>=0; li--) {
			var l = lrs[li];  //console.log(l);
			if(w==0) {
				if(l.type=="SoftmaxWithLoss" || l.type=="ReLU") {}
				else if(l.type=="InnerProduct")  dlen=l.blob[0].shape.dim[1];
				else if(l.type=="Pooling"    ) {  var prm=l.pooling_param    , ks=prm.kernel_size;  if(ks!=prm.stride) throw "e";  dlen*=ks*ks;   }
				else if(l.type=="Convolution") {  var prm=l.convolution_param, ks=prm.kernel_size[0];  if(prm.stride!=1) throw "e";  
												  var dim=l.blob[0].shape.dim;  d=dim[1];  w=h=Math.sqrt(dlen/dim[0])+ks-1;  dlen=0;  }
				else throw "e";
			}
			else {
				if     (l.type=="Data") {}
				else if(l.type=="Pooling") {  var prm=l.pooling_param, ks=prm.kernel_size;  if(ks!=prm.stride) throw "e";  w*=ks;  h*=ks;   }
				else if(l.type=="Convolution") {  var prm=l.convolution_param, ks=prm.kernel_size[0];  if(prm.stride!=1) throw "e";  
												 var dim=l.blob[0].shape.dim;  w=w+ks-1;  h=h+ks-1;  d=dim[1];   }
				else throw l.type;
			}
		}
		
		var out = [{typ:["inpt","line",w, h, d], ws:[]}];  // building a network
		for(var li=0; li<lrs.length; li++) {
			var l = lrs[li];
			if(l.type=="Data") {}
			else if(l.type=="Convolution") {
				var dim = l.blob[0].shape.dim, ks=dim[2], k2=ks*ks*dim[1], ws=[];
				w = w-ks+1;  h=h-ks+1;  d=dim[0];
				for(var oz=0; oz<d; oz++) {
					var arr=[];  ws.push(arr);
					for(var x=0; x<k2; x++) arr[x] = l.blob[0].data[oz*k2+x];
					arr[k2]=l.blob[1].data[oz];
				}
				out.push({typ:["conv","line",w,h,d, ks,1,0],ws:ws});
			}
			else if(l.type=="Pooling") {
				var prm = l.pooling_param, ks=prm.kernel_size;
				w=w/ks;  h=h/ks;
				out.push({typ:["pool","line",w,h,d, ks,ks],ws:[]});
			}
			else if(l.type=="InnerProduct") {
				var dim = l.blob[0].shape.dim, ws=[];  //console.log(l);
				w=dim[0];  h=1;  d=1;
				for(var ox=0; ox<w; ox++) {
					var arr=[];  ws.push(arr);
					for(x=0; x<dim[1]; x++) arr[x]=l.blob[0].data[ox*dim[1]+x];
					arr[dim[1]]=l.blob[1].data[ox];
				}
				out.push({typ:["full","line",w,h,d],ws:ws});
			}
			else if(l.type=="ReLU" || l.type=="SoftmaxWithLoss") {
				out[out.length-1].typ[1]={"ReLU":"relu","SoftmaxWithLoss":"sfmx"}[l.type];
			}
			else throw l.type;
		}
		return out;
	}
	UNN.util.schemes = {
		"NetParameter"  : {1:["string","name",0],100:["LayerParameter","layer"]},  // NetParameter
		"LayerParameter": {1:["string","name",0],  2:["string","type",0],3:["string","bottom"],4:["string","top"],
						   5:["float","loss_weight"],6:["ParamSpec","param"],7:["BlobProto","blob"],8:["NetStateRule","include"],10:["Phase","phase",0],
						   100:["TransformationParameter","transform_param",0],106:["ConvolutionParameter","convolution_param",0],107:["DataParameter","data_param",0],
						   108:["DropoutParameter","dropout_param",0],
						   117:["InnerProductParameter","inner_product_param",0],121:["PoolingParameter","pooling_param",0]},  
		"NetStateRule"  : {1:["Phase","phase",1]},
		"TransformationParameter" : {1:["float","scale",0],2:["bool","mirror",0],3:["uint32","crop_size",0],5:["float","mean_value"]},
		"DataParameter": {1:["string","source",0],4:["uint32","batch_size",0],8:["DB","backend",0]},
		"ParamSpec": {3:["float","lr_mult",0]},
		"ConvolutionParameter":{1:["uint32","num_output",0],4:["uint32","kernel_size"],6:["uint32","stride"],7:["FillerParameter","weight_filler",0],
								8:["FillerParameter","bias_filler",0]},
		"PoolingParameter":{1:["PoolMethod","pool",0],2:["uint32","kernel_size",0],3:["uint32","stride",0]},
		"FillerParameter":{1:["string","type",0],5:["float","mean",0],6:["float","std",0]},
		"InnerProductParameter":{1:["uint32","num_output",0],3:["FillerParameter","weight_filler",0],4:["FillerParameter","bias_filler",0]},
		"DropoutParameter":{1:["float","dropout_ratio",0]},
		"BlobProto": {5:["float","data",0],7:["BlobShape","shape",0]},
		"BlobShape": {1:["int64","dim",0]}
	};
	UNN.util.parseProtobuf = function(data,depth,schi) {
		var off = 0;
		var readVarint = function(d) {  var n=0, i=0;  while(d[off]>127) {  n=n|((d[off]&127)<<(i*7));  off++;  i++;  }  n=n|((d[off]&127)<<(i*7));  off++;  return n;  }
		var readASCII = function(d,l) {  var s="";  for(var i=0; i<l; i++) s+=String.fromCharCode(d[off+i]);  return s;  }
		
		var schema = UNN.util.schemes[schi];
		var out={}, prf="";  for(var i=0; i<depth; i++) prf+="    ";
		
		while(off<data.length) {
			var b = readVarint(data), type=b&7, fn=b>>>3;
			var tag = (schema&&schema[fn])?schema[fn]:["---",":::"+fn];
			var ptyp = tag[0], name = tag[1];
			
			var val = null;
			if(false) {}
			else if(type==0) {  val = readVarint(data);  }
			else if(type==1) {  off+=8;  }
			else if(type==2) {
				var l = readVarint(data);
				var pUP=ptyp[0].toUpperCase(), pLOW=ptyp[0].toLowerCase();
				
				if(ptyp=="string") val=readASCII(data,l);
				else if(ptyp=="float" ) val = new Float32Array(data.slice(off,off+l).buffer);
				else if(ptyp=="double") val = new Float64Array(data.slice(off,off+l).buffer);
				else if(ptyp=="int64" ) {  val = [];  for(var i=0; i<l; i++) val[i]=readVarint(data);  }
				else if(ptyp[0]==pUP && pLOW!=pUP){
					//console.log(prf, name+":", ptyp);
					var ndata = new Uint8Array(data.buffer, data.byteOffset+off,l);
					val = UNN.util.parseProtobuf(ndata,depth+1,ptyp);  //if(depth==0) throw "e";
				}
				else throw tag+" of "+schi;
				off+=l;
			}
			else if(type==5) {
				if(ptyp=="float") val = new Float32Array(data.slice(off,off+4).buffer)[0];
				else throw tag;
				off+=4;
			}
			else throw tag;
			
			//var tval = val;  if(val && val.buffer) tval = (val.length<5 ? val.join(",") : "Array("+val.length+")");
			//if(!(val instanceof Object)) console.log(prf, name+":", tval);
			if(tag[2]==0) out[name]=val;
			else {  if(out[name]==null) out[name]=[];  out[name].push(val);  }
		}
		return out;
	}
	
	