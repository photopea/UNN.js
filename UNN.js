	
	var UNN = {};

	UNN.Array = Array;

	UNN.func = {
		Lin  : function(a) {  return a;  },
		DLin : function(a) {  return 1;  },
		Sigm : function(a) { return 1 / (1 + Math.exp(-a)); },
		DSigm: function(a) {  return a*(1-a);  },
		Relu : function(a) {  return a< 0?0:a;  },
		DRelu: function(a) {  return a<=0?0:1;  },
		Tanh : function(a) {  var y = Math.exp(2*a);  return (y-1) / (y+1);  },
		DTanh: function(a) {  return 1 - a*a;  }
		//ASigm: function(a) { return a<-10 ? 0 : a>9.9 ? 1 : NN._arr[~~((a+10)*100)] }  // not precise, converges slower
	};
	
	//NN._arr = [];  for(var i=0; i<2000; i++) NN._arr.push(NN.func.Sigm(-10+i*0.01));
	
	UNN.math = {
		//Dot  : function(a, b) { var l=a.length, r=a[0]*b[0];  for(var i=1; i<l; i+=4) r += a[i]*b[i]+a[i+1]*b[i+1]+a[i+2]*b[i+2]+a[i+3]*b[i+3];  return r; },
		DotM : function(a, b) { var l=a.length, r=b[l], i=0;  while(((l-i)&3)!=0) r+=a[i]*b[i++];  for(; i<l; i+=4) r+=a[i]*b[i]+a[i+1]*b[i+1]+a[i+2]*b[i+2]+a[i+3]*b[i+3]; return r; },
		Dot  : function(a, b) { var l=a.length; var r = 0; for(var i=0; i<l; i++) r += a[i] * b[i]; return r; }
	};
	
	
	// Layers 0, 1, 2, 3 ....
	// net[0]: for each neuron in layer 1, contains array of weights to previous neurons
	UNN.size = function(typ,pa) {  return (typ[2]+2*pa)*(typ[3]+2*pa)*typ[4];  }
	
	UNN.Create = function(topo, v) {
		var net = [];
		for(var i=0; i<topo.length; i++) {
			var typ=topo[i], ptyp=i==0 ? 0 : topo[i-1];
			var csz=UNN.size(typ,0), psz=UNN.size(ptyp,0);
			var arr = [];  net.push({ typ:typ, ws: arr });
			if(typ[0]=="full") {
				var pcnt = psz+1;
				for(var j=0; j<csz; j++) {
					var syn = new UNN.Array(pcnt);  arr.push(syn);
					for(var k=0; k<pcnt; k++) syn[k] = -v+Math.random()*2*v;
				}
			}
			else if(typ[0]=="conv") {
				var K = typ[5], S=typ[6], P=typ[7], pcnt = K*K*ptyp[4]+1;
				var nx = (ptyp[2]-K+2*P)/S + 1;
				var ny = (ptyp[3]-K+2*P)/S + 1;
				if(nx!=typ[2] || ny!=typ[3]) throw "["+ptyp+"] => ["+typ+"]";
				for(var d=0; d<typ[4]; d++) {
					var syn = new UNN.Array(pcnt);  arr.push(syn);
					for(var k=0; k<pcnt; k++) syn[k] = -v+Math.random()*2*v;
				}
			}
			else if(typ[0]=="pool" || typ[0]=="inpt") {}
			else throw typ;
		}
		return net;
	}
	
	UNN.GetOutput = function(Ls, Inpt, Os) {
		var il = Inpt.length;
		if(Os.length==0) 
			for(var l=0; l<Ls.length; l++) {
				var pa = 0;  if(l+1<Ls.length && Ls[l+1].typ[0]=="conv") pa=Ls[l+1].typ[7];
				var len  = UNN.size(Ls[l].typ,0 );
				var plen = UNN.size(Ls[l].typ,pa);
				Os[l]=[new UNN.Array(len),[],pa==0 ? null : new UNN.Array(plen)];
			}
			
		var FOs = Os[0][0];
		for(var i=0; i<il; i++) FOs[i]=Inpt[i];
		
		for(var l=1; l<Ls.length; l++) {
			var Lay = Ls[l], La0=Ls[l-1], Wsl = Lay.ws, len=Wsl.length;
			var Pi = Os[l][0], Ii=Os[l][1], Is=Os[l-1][0];
			if(Lay.typ[0]=="full") {
				for(var ni=0; ni<len; ni++) {
					var ksi  = UNN.math.DotM(Is, Wsl[ni]);
					//var ksi2 = Wsl[ni][Os[l].length] + NN.math.Dot(Os[l], Wsl[ni]);  if(Math.abs(ksi-ksi2)>1e-7) {  console.log(ksi,ksi2);  throw "e";  }
					Pi[ni] = ksi;
				}
			}
			else if(Lay.typ[0]=="conv") {
				var w =Lay.typ [2], h =Lay.typ [3], d =Lay.typ [4], ks=Lay.typ[5], st=Lay.typ[6], pa=Lay.typ[7];
				var pw=La0.typ [2], ph=La0.typ [3], pd=La0.typ [4];
				
				if(pa!=0) {
					var nw = (pw+2*pa), nh=(ph+2*pa);
					var NIs =Os[l-1][2];  NIs.fill(0);
					for(var z=0; z<pd; z++)
						for(var y=0; y<ph; y++) {
							var so = pw*ph*z + y*pw, to = nw*nh*z + (pa+y)*nw + pa;
							for(var x=0; x<pw; x++) NIs[to+x] = Is[so+x];
						}
					Is = NIs;  pw=nw;  ph=nh;
				}
				
				for(var z=0; z<d; z++) {
					var Ke=Lay.ws[z], bi=Ke[ks*ks*pd];  // bias
					//if(Ke.length!=ks*ks+1) {  console.log(Ke,ks);  throw "e";  }
					if(ks==3) 
					for(var y=0; y<h; y++) {
						var oo = z*h*w + y*w;
						for(var x=0; x<w; x++) {
							var ksi = bi, ko=0;
							for(var kz=0; kz<pd; kz++) {
								var io = kz*pw*ph + (y*st)*pw + x*st;
								ksi += Ke[ko+0]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2];  io+=pw;
								ksi += Ke[ko+3]*Is[io] + Ke[ko+4]*Is[io+1] + Ke[ko+5]*Is[io+2];  io+=pw;
								ksi += Ke[ko+6]*Is[io] + Ke[ko+7]*Is[io+1] + Ke[ko+8]*Is[io+2];  ko+=9;  if(Is[io+2]==null) throw "e";
							}
							Pi[oo + x] = ksi;
						}
					}
					else if(ks==5)
					for(var y=0; y<h; y++) {
						var oo = z*h*w + y*w;
						for(var x=0; x<w; x++) {
							var ksi = bi, ko=0;
							for(var kz=0; kz<pd; kz++) {
								var io = kz*pw*ph + (y*st)*pw + x*st;
								ksi += Ke[ko]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2] + Ke[ko+3]*Is[io+3] + Ke[ko+4]*Is[io+4];  io+=pw;  ko+=5;
								ksi += Ke[ko]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2] + Ke[ko+3]*Is[io+3] + Ke[ko+4]*Is[io+4];  io+=pw;  ko+=5;
								ksi += Ke[ko]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2] + Ke[ko+3]*Is[io+3] + Ke[ko+4]*Is[io+4];  io+=pw;  ko+=5;
								ksi += Ke[ko]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2] + Ke[ko+3]*Is[io+3] + Ke[ko+4]*Is[io+4];  io+=pw;  ko+=5;
								ksi += Ke[ko]*Is[io] + Ke[ko+1]*Is[io+1] + Ke[ko+2]*Is[io+2] + Ke[ko+3]*Is[io+3] + Ke[ko+4]*Is[io+4];  ko+=5;
							}
							Pi[oo + x] = ksi;
						}
					}
					else 
					for(var y=0; y<h; y++) {
						var oo = z*h*w + y*w;
						for(var x=0; x<w; x++) {
							var ksi = bi;
							for(var kz=0; kz<pd; kz++) {
								var io = kz*pw*ph + (y*st)*pw + x*st, ko=kz*ks*ks;
								for(var ky=0; ky<ks; ky++) {
									for(var kx=0; kx<ks; kx++) ksi += Ke[ko+kx] * Is[io+kx];
									io+=pw;  ko+=ks;
								}
							}
							Pi[oo + x] = ksi;
						}
					}
				}
			}
			else if(Lay.typ[0]=="pool") {
				var w =Lay.typ [2], h =Lay.typ [3], d =Lay.typ [4], ks=Lay.typ[5], st=Lay.typ[6];
				var pw=La0.typ [2], ph=La0.typ [3], pd=La0.typ [4];
				
				if(ks==2)
				for(var z=0; z<d; z++)
					for(var y=0; y<h; y++) {
						var oo = z*h*w + y*w;
						var io = z*pw*ph + (y*st)*pw;
						for(var x=0; x<w; x++) {
							var mi=io, mv = Is[mi], ci=0, cv=0;
							ci=io+1   ;  cv=Is[ci];  if(cv>mv) {  mi=ci;  mv=cv;  }
							ci=io+pw  ;  cv=Is[ci];  if(cv>mv) {  mi=ci;  mv=cv;  }
							ci=io+pw+1;  cv=Is[ci];  if(cv>mv) {  mi=ci;  mv=cv;  }
							Pi[oo + x] = mv;  Ii[oo + x] = mi;  io+=st;
						}
					}
				else
				for(var z=0; z<d; z++)
					for(var y=0; y<h; y++) {
						var oo = z*h*w + y*w;
						for(var x=0; x<w; x++) {
							var io = z*pw*ph + (y*st)*pw + x*st;
							var mi=0, mv = -1e9;
							for(var ky=0; ky<ks; ky++) {
								for(var kx=0; kx<ks; kx++) {  var cv = Is[io+kx];  if(cv>mv) {  mv=cv;  mi=io+kx;  }  }
								io+=pw;
							}
							Pi[oo + x] = mv;  Ii[oo + x] = mi;
						}
					}
				
			}
			else throw "e";
			
			var pl = Pi.length;
			if     (Lay.typ[1]=="sigm") for(var i=0; i<Pi.length; i++) Pi[i]=UNN.func.Sigm(Pi[i]);
			else if(Lay.typ[1]=="tanh") for(var i=0; i<Pi.length; i++) Pi[i]=UNN.func.Tanh(Pi[i]);
			else if(Lay.typ[1]=="relu") for(var i=0; i<Pi.length; i++) Pi[i]=UNN.func.Relu(Pi[i]);
			else if(Lay.typ[1]=="line") {}
			else if(Lay.typ[1]=="sfmx") {
				var len = Pi.length;
				var amax = Pi[0], esum=0;
				for(var i=1;i<len;i++) if(Pi[i]>amax) amax = Pi[i];
				for(var i=0;i<len;i++) {  var e=Math.exp(Pi[i]-amax);  esum+=e;  Pi[i]=e;  }
				var isum = 1/esum;
				for(var i=0;i<len;i++) Pi[i] *= isum;
			}
			else throw Lay.typ[1];
		}
	}
	
	UNN.GetError = function(Ws, Ps, Ds) {	
		var err = 0;
		for(var i0=0; i0<Ps.length; i0++) {
			var D = Ds[i0], O=[];
			UNN.GetOutput(Ws, Ps[i0], O);
			var Y = O.pop();
			for(var i=0; i<D.length; i++) { var df = Y[i]-D[i]; err += df*df; }
		}
		return 0.5*err/Ps.length;
	}
	
	UNN.GetDeltas = function(Ls, Os, Rslt, Deltas) {
		
		if(Deltas.length==0) for(var l=1; l<Os.length; l++) Deltas.push( [new UNN.Array(Os[l][0].length),[],Os[l][2]?new UNN.Array(Os[l][2].length):null] );
		
		for(var l=Ls.length-1; l>0; l--) {
			var Dlt = Deltas[l-1][0], cnt = Dlt.length;
			var Oss=Os[l][0];
			
			if(l==Ls.length-1)
				for(var j=0; j<cnt; j++) Dlt[j] = Rslt[j]-Oss[j];
			else {
				for(var j=0; j<cnt; j++) Dlt[j]=0;
				
				var Dl2 = Deltas[l][0], len=Dl2.length, Lay = Ls[l+1], La0=Ls[l], typ=Lay.typ[0], Wl2 = Lay.ws;
				
				if(typ=="full") {					
					for(var k=0; k<len; k++) { // for each neuron in the next layer
						var Wlk=Wl2[k], dlk=Dl2[k];
						// to every current neuron Delta, add a multiple of weight and next Delta
						for(var j=0; j<cnt; j++)  Dlt[j] += dlk * Wlk[j];
					}
				}
				else if(typ=="conv") {
					//console.log(Dlt);  throw "e";
					//throw "e";
					var w =Lay.typ [2], h =Lay.typ [3], d =Lay.typ [4], ks=Lay.typ[5], st=Lay.typ[6], pa=Lay.typ[7];
					var pw=La0.typ [2], ph=La0.typ [3], pd=La0.typ [4];
					
					if(pa!=0) {  Dlt=Deltas[l-1][2];  pw+=2*pa;  ph+=2*pa;  var dl=Dlt.length; for(var i=0;i<dl;i++) Dlt[i]=0;  }
					
					for(var z=0; z<d; z++) {
						var Ke=Lay.ws[z];
						if(ks==3) 
						for(var y=0; y<h; y++)
							for(var x=0; x<w; x++) {
								var dlk = Dl2[(z*w*h) + y*w + x]
								var dx = x*st, dy=y*st, ko=0;
								for(var kz=0; kz<pd; kz++) {
									var io = kz*pw*ph + dy*pw + dx;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  io+=pw;  ko+=3;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  io+=pw;  ko+=3;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  ko+=3;
								}
							}
						else if(ks==5) 
						for(var y=0; y<h; y++)
							for(var x=0; x<w; x++) {
								var dlk = Dl2[(z*w*h) + y*w + x]
								var dx = x*st, dy=y*st, ko=0;
								for(var kz=0; kz<pd; kz++) {
									var io = kz*pw*ph + dy*pw + dx;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  Dlt[io+3]+=dlk*Ke[ko+3];  Dlt[io+4]+=dlk*Ke[ko+4];  io+=pw;  ko+=5;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  Dlt[io+3]+=dlk*Ke[ko+3];  Dlt[io+4]+=dlk*Ke[ko+4];  io+=pw;  ko+=5;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  Dlt[io+3]+=dlk*Ke[ko+3];  Dlt[io+4]+=dlk*Ke[ko+4];  io+=pw;  ko+=5;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  Dlt[io+3]+=dlk*Ke[ko+3];  Dlt[io+4]+=dlk*Ke[ko+4];  io+=pw;  ko+=5;
									Dlt[io]+=dlk*Ke[ko];  Dlt[io+1]+=dlk*Ke[ko+1];  Dlt[io+2]+=dlk*Ke[ko+2];  Dlt[io+3]+=dlk*Ke[ko+3];  Dlt[io+4]+=dlk*Ke[ko+4];  ko+=5;
								}
							}
						else 
						for(var y=0; y<h; y++)
							for(var x=0; x<w; x++) {
								var dlk = Dl2[(z*w*h) + y*w + x]
								var dx = x*st, dy=y*st, ko=0;
								for(var kz=0; kz<pd; kz++) {
									var io = kz*pw*ph + dy*pw + dx;
									for(var ky=0; ky<ks; ky++) {
										for(var kx=0; kx<ks; kx++) {
											if(Dlt[io+kx]==null) {  console.log(kz,pw,ph,dy,pw,dx, Dlt, io+kx);  throw "e";  }
											Dlt[io+kx] += dlk * Ke[ko + kx];
										}
										io += pw;  ko+=ks;
									}
								}
							}
					}
					
					if(pa!=0) {
						var ODlt = Deltas[l-1][0], ow=pw-2*pa, oh=ph-2*pa;
						for(var z=0; z<pd; z++)
							for(var y=0; y<oh; y++) {
								var so = pw*ph*z + (y+pa)*pw + pa, to = ow*oh*z + y*ow;
								for(var x=0; x<ow; x++) ODlt[to+x] = Dlt[so+x];
							}
					}
					
					//console.log(Dlt);  throw "e";
				}
				else if(typ=="pool") {
					var Ii = Os[l+1][1];
					for(var k=0; k<len; k++) Dlt[Ii[k]] = Dl2[k];
				}
			}
			
			var afun = Ls[l].typ[1];
			if     (afun=="sigm") for(var i=0; i<cnt; i++)  Dlt[i] *= UNN.func.DSigm(Oss[i]);
			else if(afun=="tanh") for(var i=0; i<cnt; i++)  Dlt[i] *= UNN.func.DTanh(Oss[i]);
			else if(afun=="relu") for(var i=0; i<cnt; i++)  Dlt[i] *= UNN.func.DRelu(Oss[i]);
			else if(afun=="line") {}
			else if(afun=="sfmx") {}
			else throw afun;
			// Should not be done for "pool" layers
			//if(l==Ls.length-1  ||  Ls[l+1].typ[0]!="pool")  // same as checking .ptyp[0]
			//	for(var i=0; i<cnt; i++)  Dlt[i] *= DAF(Oss[i]);
		}
		return Deltas;
	}
	UNN.opt = {
		sgd     : function(gt,g2j,d2j,i) {  
			return gt;
		},
		momentum: function(gt,g2j,d2j,i) {  
			var dif = g2j[i] = 0.05*gt + 0.95*g2j[i];  return dif;
		},
		adagrad : function(gt,g2j,d2j,i) {
			g2j[i]+=gt*gt;
			//if(isNaN(g2j[i]))throw "e";
			return gt * 0.01/Math.sqrt(g2j[i] + 1e-6);
		},
		adadelta: function(gt,g2j,d2j,i) {
			var rho = 0.95, eps = 1e-6;
			g2j[i] = rho*g2j[i] + (1-rho)*(gt*gt);
			var df = gt*Math.sqrt((d2j[i]+eps)/(g2j[i]+eps));
			d2j[i] = rho*d2j[i] + (1-rho)*(df*df);
			return df;
		}
	}
	
	UNN.clone = function(W) {
		var out = [];
		for(var i=0; i<W.length; i++) {
			var lay = W[i], nlay={typ:lay.typ.slice(0), ws:[]};  out.push(nlay);
			for(var j=0; j<lay.ws.length; j++) nlay.ws.push(lay.ws[j].slice(0));
		}
		return out;
	}
	UNN._zerofy = function(W) {
		for(var i=0; i<W.length; i++) {  var ws=W[i].ws;  for(var j=0; j<ws.length; j++) for(var k=0; k<ws[j].length; k++)  ws[j][k]=0;  }
	}
	
	UNN.Train = function(W, Ins, Outs, prm, debug)
	{
		//var nW = W.slice(1);
		var N = Ins.length, BS=prm.batch_size, iBS=1/BS, alim = 0//0.0000000001;  // does it influence learning?
		var Os = [], Deltas = [], err = 0;
		var isq = 1/Math.sqrt(1+prm.itr);
		var pow = Math.pow(0.8, prm.itr);
		//var as = {"sigm":0.2, "tanh":0.018, "relu":0.01, "line":0.1, "sfmx":0.2};
		
		var Acc = UNN.clone(W), Eg2 = UNN.clone(W), Ed2 = UNN.clone(W);
		UNN._zerofy(Acc);  UNN._zerofy(Eg2);  UNN._zerofy(Ed2);
		//for(var i=0; i<W.length; i++)	for(var j=0; j<W[i].ws.length; j++) for(var k=0; k<W[i].ws[j].length; k++) {  Eg2[i].ws[j][k]=Ed2[i].ws[j][k]=0;  }
		
		var opt = UNN.opt[prm.method];
		
		for(var p=0; p<N; p++) {
			if(debug && (p%10000)==0) console.log(p,N);
			var I = Ins[p];
			var O = Outs[p];
			
			UNN.GetOutput(W, I, Os);
			
			var Y = Os[Os.length-1][0];
			for(var i=0; i<O.length; i++) { var df = Y[i]-O[i]; err += df*df; }
			UNN.GetDeltas(W, Os, O, Deltas);
			//continue;
			for(var l=1; l<W.length; l++) {      // for each neuron layer
				var Lay = W[l], La0=W[l-1], Wl=Acc[l].ws, cnt=Wl.length, Osl=Os[l-1][0], Dl=Deltas[l-1][0], pcnt = Osl.length;  //if((pcnt&3)!=0) throw "e";
				var a = 1;
				if(Lay.typ[0]=="full") 
					for(var j=0; j<cnt; j++) { // for each neuron
						var ad = a*Dl[j];
						if(ad*ad>alim) {
							var Wlj=Wl[j], i=0;
							Wlj[pcnt] += ad;
							
							// for each input to that neuron
							//for(; i<pcnt; i++)  Wlj[i] += ad*Osl[i];
							while(((pcnt-i)&3)!=0) {  Wlj[i]+=ad*Osl[i];  i++;  }
							for(; i<pcnt; i+=4) {  Wlj[i  ]+=ad*Osl[i  ];  Wlj[i+1]+=ad*Osl[i+1];  
												   Wlj[i+2]+=ad*Osl[i+2];  Wlj[i+3]+=ad*Osl[i+3];  }
						}
					}
				else if(Lay.typ[0]=="conv") {
					var w =Lay.typ [2], h =Lay.typ [3], d =Lay.typ [4], ks=Lay.typ[5], st=Lay.typ[6], pa=Lay.typ[7];
					var pw=La0.typ [2], ph=La0.typ [3], pd=La0.typ [4], kel=ks*ks*pd;
					
					if(pa!=0) {  Osl = Osl=Os[l-1][2];  pw+=2*pa;  ph+=2*pa;  }
					
					for(var z=0; z<d; z++) {
						var Ke=Wl[z];
						if(ks==3)
						for(var y=0; y<h; y++) {
							var oo = (z*w*h)+y*w;
							for(var x=0; x<w; x++) {  // for each output neuron
								var ad = a*Dl[oo+x];  //if(Dl[(z*w*h)+y*w+x]==null) {  console.log((z*w*h)+y*w+x, Lay,Dl);  throw "e";  }
								if(ad*ad>alim) {
									Ke[kel] += ad;
									for(var kz=0; kz<pd; kz++) {
										var io = kz*pw*ph + (y*st)*pw + x*st, ko=kz*9;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  io+=pw; ko+=3;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  io+=pw; ko+=3;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  io+=pw; ko+=3;
									}
								}
							}
						}
						else if(ks==5)
						for(var y=0; y<h; y++) {
							var oo = (z*w*h)+y*w;
							for(var x=0; x<w; x++) {  // for each output neuron
								var ad = a*Dl[oo+x];  //if(Dl[(z*w*h)+y*w+x]==null) {  console.log((z*w*h)+y*w+x, Lay,Dl);  throw "e";  }
								if(ad*ad>alim) {
									Ke[kel] += ad;
									for(var kz=0; kz<pd; kz++) {
										var io = kz*pw*ph + (y*st)*pw + x*st, ko=kz*25;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  Ke[ko+3]+=ad*Osl[io+3];  Ke[ko+4]+=ad*Osl[io+4];  io+=pw; ko+=5;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  Ke[ko+3]+=ad*Osl[io+3];  Ke[ko+4]+=ad*Osl[io+4];  io+=pw; ko+=5;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  Ke[ko+3]+=ad*Osl[io+3];  Ke[ko+4]+=ad*Osl[io+4];  io+=pw; ko+=5;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  Ke[ko+3]+=ad*Osl[io+3];  Ke[ko+4]+=ad*Osl[io+4];  io+=pw; ko+=5;
										Ke[ko]+=ad*Osl[io];  Ke[ko+1]+=ad*Osl[io+1];  Ke[ko+2]+=ad*Osl[io+2];  Ke[ko+3]+=ad*Osl[io+3];  Ke[ko+4]+=ad*Osl[io+4];  //io+=pw; ko+=5;
									}
								}
							}
						}
						else 
						for(var y=0; y<h; y++) {
							var oo = (z*w*h)+y*w;
							for(var x=0; x<w; x++) {  // for each output neuron
								var ad = a*Dl[oo+x];  //if(Dl[(z*w*h)+y*w+x]==null) {  console.log((z*w*h)+y*w+x, Lay,Dl);  throw "e";  }
								if(ad*ad>alim) {
									Ke[kel] += ad;
									for(var kz=0; kz<pd; kz++) {
										var io = kz*pw*ph + (y*st)*pw + x*st, ko=kz*ks*ks;
										for(var ky=0; ky<ks; ky++) {
											for(var kx=0; kx<ks; kx++) {
												Ke[ko+kx] += ad*Osl[io+kx];
											}
											io+=pw;  ko+=ks;
										}
									}
								}
							}
						}	
					}
				}
			}
			if((p%BS)==BS-1) {
				for(var l=1; l<W.length; l++) {      // for each neuron layer
					if(W[l].typ[0]!="pool") {
						var Wl=W[l].ws, Al = Acc[l].ws, cnt=Al.length, pcnt = Wl[0].length;  //if((pcnt&3)!=0) throw "e";
						var ad = iBS//as[Lay.typ[1]]*pow;
						//var ad = a;
						var g2 = Eg2[l].ws, d2 = Ed2[l].ws;
						for(var j=0; j<cnt; j++) { // for each neuron
							var Wlj = Wl[j], Alj = Al[j], i=0;
							var g2j = g2[j], d2j = d2[j];
							// for each input to that neuron
							//for(; i<pcnt; i++)  {  Wlj[i] += opt(ad*Alj[i],g2j,d2j,i);  Alj[i]=0;  }
								
							while(((pcnt-i)&3)!=0) {  Wlj[i]+=opt(ad*Alj[i],g2j,d2j,i);  Alj[i]=0;  i++;  }
							for(; i<pcnt; i+=4) {  Wlj[i  ]+=opt(ad*Alj[i  ],g2j,d2j,i  );  Alj[i  ]=0;  Wlj[i+1]+=opt(ad*Alj[i+1],g2j,d2j,i+1);  Alj[i+1]=0; 
												   Wlj[i+2]+=opt(ad*Alj[i+2],g2j,d2j,i+2);  Alj[i+2]=0;  Wlj[i+3]+=opt(ad*Alj[i+3],g2j,d2j,i+3);  Alj[i+3]=0; }
						}
					}
				}
			}
		}
			
		return 0.5*err/N;
	}
	