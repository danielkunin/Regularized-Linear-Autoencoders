if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var renderer, scene, camera;

init();
animate();


function init() {

	var container = document.getElementById( 'container' );

	scene = new THREE.Scene();
	scene.background = new THREE.Color( "white" );

	camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 1, 1000 );
	camera.position.set( -25, -25, 20 );
	camera.up.set( 0, 0, 1 );

	var group = new THREE.Group();
	scene.add( group );

	var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.6 );
	directionalLight.position.set( 0.75, 0.75, 1.0 ).normalize();
	scene.add( directionalLight );

	var ambientLight = new THREE.AmbientLight( 0xcccccc, 0.2 );
	scene.add( ambientLight );

	var helper = new THREE.GridHelper( 20, 10 );
	helper.rotation.x = Math.PI / 2;
	helper.position.set( 0, 0, 0 );
	group.add( helper );

	var w1 = makeTextSprite( "w1", { fontsize: 20 } );
	var w2 = makeTextSprite( "w2", { fontsize: 20 } );
	w1.position.set(0,-12,0);
	w2.position.set(-12,0,0);
	group.add( w1 );
	group.add( w2 );

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	container.appendChild( renderer.domElement );

	controls = new THREE.OrbitControls( camera, renderer.domElement );

	// Comment to stop rotation
	// controls.autoRotate = true;

	window.addEventListener( 'resize', onWindowResize, false );

}


function onWindowResize() {

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

	renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {

	requestAnimationFrame( animate );
	var dist = (camera.position.x**2 + camera.position.y**2 + camera.position.z**2)**0.25;
	camera.lookAt(0,0,dist) 

	// Comment to stop rotation
	// controls.update();

	render();

}

function render() {

	renderer.render( scene, camera );

}


var geometry, object, material;

function removeLandscape() {

	scene.remove(object);
	if (object != undefined) {
		object.geometry.dispose();
		object.material.dispose();
		object = undefined;
	} 

}

function addLandscape(loss, x, param) {

	material = new THREE.MeshPhongMaterial( {
					color: 0x156289,
					emissive: 0x072534,
					side: THREE.DoubleSide,
					flatShading: true,
					transparent: true,
					opacity: 0.8
				} );
	geometry = new THREE.ParametricBufferGeometry( THREE.ParametricGeometries[loss](x, param, 4,4), 75, 75 );
	object = new THREE.Mesh( geometry, material );
	object.position.set( 0, 0, 0 );
	object.scale.multiplyScalar( 5 );

	scene.add( object );
}


THREE.ParametricGeometries = {

	Unregularized: function ( data, param, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var x = data[0]
			var y = data[1]
			var tied = param[2]

			if (tied) {
				var z = ((1 - w1**2)**2 + (w1*w2)**2) * x + ((1 - w2**2)**2 + (w1*w2)**2) * y;
			} else {
				var z = (y - w2 * w1 * x)**2;
			}

			target.set( w1, w2, z );
		};
	},

	Product: function ( data, param, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var x = data[0]
			var y = data[1]
			var lamb = param[0]
			var pow = param[1]
			var tied = param[2]

			if (tied) {
				var z = ((1 - w1**2)**2 + (w1*w2)**2) * x + ((1 - w2**2)**2 + (w1*w2)**2) * y + lamb * (Math.abs(w1)**(2*pow) + 2*Math.abs(w1 * w2)**pow + Math.abs(w2)**(2*pow));
			} else {
				var z = (y - w2 * w1 * x)**2 + lamb * Math.abs(w2 * w1)**pow;
			}

			target.set( w1, w2, z );
		};
	},

	Sum: function ( data, param, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var x = data[0]
			var y = data[1]
			var lamb = param[0]
			var pow = param[1]
			var tied = param[2]

			if (tied) {
				var z = ((1 - w1**2)**2 + (w1*w2)**2) * x + ((1 - w2**2)**2 + (w1*w2)**2) * y + 2 * lamb * (Math.abs(w1)**pow + Math.abs(w2)**pow);
			} else {
				var z = (y - w2 * w1 * x)**2 + lamb * (Math.abs(w1)**pow + Math.abs(w2)**pow);
			}

			target.set( w1, w2, z );
		};
	},

	alignment: function ( data, param, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var x = data[0]
			var y = data[1]
			var lamb0 = param[0]
			var lamb1 = param[1]
			var lamb2 = param[2]
			var lamb3 = param[3]
			var z = lamb0 * (y - w1 * x)**2 + lamb1 * (x - w2 * w1 * x)**2 + lamb2 * (w1**2 + w2**2) - 2 * lamb3 * (w2 * w1);
			target.set( w1, w2, z );
		};
	}

};



var scalar = function() {
  this.loss = 'Unregularized',
  this.x = 0.25,
  this.lamb = 0.25,
  this.pow = 2
  this.tied = false
};

var vector = function() {
  this.loss = 'Unregularized',
  this.x = 0.25,
  this.y = 0.25,
  this.lamb = 0.25,
  this.pow = 2
  this.tied = false
};	

var alignment = function() {
  this.loss = 'alignment',
  this.x = 0.25,
  this.y = 0.25,
  this.lamb0 = 0.25,
  this.lamb1 = 0.25,
  this.lamb2 = 0.25,
  this.lamb3 = 0.25
};


window.onload = function() {
	var gui = new dat.GUI();
	var f1 = gui.addFolder('Linear Autoencoder (scalar case)');
	var f2 = gui.addFolder('Linear Autoencoder (vector case)');
	var f3 = gui.addFolder('Low-rank Prediction');
	var f4 = gui.addFolder('Weight Alignment');
	
	// Linear Autoencoder (scalar case)
	var obj1 = new scalar();
	var graph1 = function() {
		removeLandscape();
		addLandscape(obj1.loss, [Math.sqrt(obj1.x), Math.sqrt(obj1.x)], [obj1.lamb, obj1.pow, obj1.tied]);
		f2.close();
		f3.close();
		f4.close();
	}
	f1.add(obj1, 'loss', ['Unregularized', 'Product', 'Sum']).onChange(graph1).name('Loss Function');
	f1.add(obj1, 'x', 0, 2).onChange(graph1).name(katex.renderToString('x^2'));
	f1.add(obj1, 'lamb', 0, 2).onChange(graph1).name(katex.renderToString('\\lambda'));
	f1.add(obj1, 'pow', 0.5, 4).onChange(graph1).name(katex.renderToString('\\alpha'));

	// Linear Autoencoder (vector case)
	var obj2 = new vector();
	obj2.tied = true;
	var graph2 = function() {
		removeLandscape();
		addLandscape(obj2.loss, [obj2.x, obj2.y], [obj2.lamb, obj2.pow, obj2.tied]);
		f1.close();
		f3.close();
		f4.close();
	}
	f2.add(obj2, 'loss', ['Unregularized', 'Product', 'Sum']).onChange(graph2).name('Loss Function');
	f2.add(obj2, 'x', 0, 2).onChange(graph2).name(katex.renderToString('\\sigma_1^2'));
	f2.add(obj2, 'y', 0, 2).onChange(graph2).name(katex.renderToString('\\sigma_2^2'));
	f2.add(obj2, 'lamb', 0, 2).onChange(graph2).name(katex.renderToString('\\lambda'));
	f2.add(obj2, 'pow', 0.5, 4).onChange(graph2).name(katex.renderToString('\\alpha'));

	// Low-rank Prediction (scalar case)
	var obj3 = new vector();
	var graph3 = function() {
		removeLandscape();
		addLandscape(obj3.loss, [obj3.x, obj3.y], [obj3.lamb, obj3.pow, obj3.tied]);
		f1.close();
		f2.close();
		f4.close();
	}
	f3.add(obj3, 'loss', ['Unregularized', 'Product', 'Sum']).onChange(graph3).name('Loss Function');
	f3.add(obj3, 'x', -2, 2).onChange(graph3).name(katex.renderToString('x'));
	f3.add(obj3, 'y', -2, 2).onChange(graph3).name(katex.renderToString('y'));
	f3.add(obj3, 'lamb', 0, 2).onChange(graph3).name(katex.renderToString('\\lambda'));
	f3.add(obj3, 'pow', 0.5, 4).onChange(graph3).name(katex.renderToString('\\alpha'));

	// Weight Alignment
	var obj4 = new alignment();
	var graph4 = function() {
		removeLandscape();
		addLandscape(obj4.loss, [obj4.x, obj4.y], [obj4.lamb0, obj4.lamb1, obj4.lamb2, obj4.lamb3]);
		f1.close();
		f2.close();
		f3.close();
	}
	f4.add(obj4, 'x', -2, 2).onChange(graph4).name(katex.renderToString('x'));
	f4.add(obj4, 'y', -2, 2).onChange(graph4).name(katex.renderToString('y'));
	f4.add(obj4, 'lamb0', 0, 2).onChange(graph4).name(katex.renderToString('\\lambda_{\\text{pred}}'));
	f4.add(obj4, 'lamb1', 0, 2).onChange(graph4).name(katex.renderToString('\\lambda_{\\text{info}}'));
	f4.add(obj4, 'lamb2', 0, 2).onChange(graph4).name(katex.renderToString('\\lambda_{\\text{reg}}'));
	f4.add(obj4, 'lamb3', 0, 2).onChange(graph4).name(katex.renderToString('\\lambda_{\\text{self}}'));

	graph1();
	f1.open();
};
