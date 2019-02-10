if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var renderer, scene, camera;

init();
animate();


function init() {

	var container = document.getElementById( 'container' );

	//

	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xb0b0b0 );

	//

	camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 1, 1000 );
	camera.position.set( 0, -30, 30 );
	camera.up.set( 0, 0, 1 );

	//

	var group = new THREE.Group();
	scene.add( group );

	//

	var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.6 );
	directionalLight.position.set( 0.75, 0.75, 1.0 ).normalize();
	scene.add( directionalLight );

	var ambientLight = new THREE.AmbientLight( 0xcccccc, 0.2 );
	scene.add( ambientLight );

	//

	var helper = new THREE.GridHelper( 16, 10 );
	helper.rotation.x = Math.PI / 2;
	helper.position.set( 0, 0, 0 );
	group.add( helper );

	//

	var w1 = makeTextSprite( "w1", { fontsize: 20 } );
	var w2 = makeTextSprite( "w2", { fontsize: 20 } );
	w1.position.set(-10,0,0);
	w2.position.set(0,-10,0);
	group.add( w1 );
	group.add( w2 )

	//

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	container.appendChild( renderer.domElement );

	//

	controls = new THREE.OrbitControls( camera, renderer.domElement );

	//

	window.addEventListener( 'resize', onWindowResize, false );

}


function onWindowResize() {

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

	renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {

	requestAnimationFrame( animate );

	camera.lookAt(0,0,10) 

	render();

}

function render() {

	renderer.render( scene, camera );

}




var geometry, object, material;

function removeLandscape() {

	scene.remove(object); 

}

function addLandscape(loss, data, lamb, mode) {

	material = new THREE.MeshPhongMaterial( {
					color: 0x156289,
					emissive: 0x072534,
					side: THREE.DoubleSide,
					flatShading: true,
					transparent: true,
					opacity: 0.8
				} )


	geometry = new THREE.ParametricBufferGeometry( THREE.ParametricGeometries[loss](data, lamb, mode, 4,4), 100, 100 );
	geometry.verticesNeedUpdate = true;
	object = new THREE.Mesh( geometry, material );
	object.position.set( 0, 0, 0 );
	object.scale.multiplyScalar( 4 );

	scene.add( object );
}

THREE.ParametricGeometries = {

	unregularized: function ( x, lamb, mode, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			if (mode == "1d") {
				var z = (x[0] - w2 * w1 * x[0])**2;
			} else {
				var z = (x[1] - w1**2 * x[1])**2 + (x[2] - w2**2 * x[2])**2;
			}

			target.set( w1, w2, z );
		};
	},

	product: function ( x, lamb, mode, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			if (mode == "1d") {
				var z = (x[0] - w2 * w1 * x[0])**2 + lamb * (w2 * w1)**2;
			} else {
				var z = (x[1] - w1**2 * x[1])**2 + (x[2] - w2**2 * x[2])**2 + lamb * (w1**4 + 2 *(w1 * w2)**2 + w2**4);
			}

			target.set( w1, w2, z );
		};
	},

	sum: function ( x, lamb, mode, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;

			if (mode == "1d") {
				var z = (x[0] - w2 * w1 * x[0])**2 + lamb * (w1**2 + w2**2);
			} else {
				var z = (x[1] - w1**2 * x[1])**2 + (x[2] - w2**2 * x[2])**2 + 2 * lamb * (w1**2 + w2**2);
			}

			target.set( w1, w2, z );
		};
	}

};

var loss = "unregularized",
	data = [1, 1, 1],
	lamb = 0.5,
	mode = "1d";
addLandscape(loss, data, lamb, mode);


$("input[name=loss]").on("change", function() {
	loss = $(this).val();
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

$("#lamb").on("input", function() {
	lamb = $(this).val();
	$("#lamb-val").text(lamb);
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

$("#cases").on("change", function() {
	$(".data").toggle()
	mode = $(this).val()
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

$("#data").on("input", function() {
	data[0] = $(this).val();
	$("#data-val").text(data);
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

$("#singular1").on("input", function() {
	data[1] = $(this).val();
	$("#singular1-val").text(data);
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

$("#singular2").on("input", function() {
	data[2] = $(this).val();
	$("#singular2-val").text(data);
	removeLandscape();
	addLandscape(loss, data, lamb, mode);
});

// TO DO:
//  - Update geometry vertices rather than creating new one for updates to lamb/data
//  - Simplify jquery event listeners
//  - Add critical points
//  - Add axis labels
