if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var renderer, scene, camera;//stats

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
	group.add( helper );


	//

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	container.appendChild( renderer.domElement );

	//

	var controls = new THREE.OrbitControls( camera, renderer.domElement );

	//

	// stats = new Stats();
	// container.appendChild( stats.dom );

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

	render();
	// stats.update();

}

function render() {

	renderer.render( scene, camera );

}




var geometry, object, material;

function removeLandscape() {

	scene.remove(object); 

}

function addLandscape(loss, data, lamb) {

	material = new THREE.MeshPhongMaterial( {
					color: 0x156289,
					emissive: 0x072534,
					side: THREE.DoubleSide,
					flatShading: true,
					transparent: true,
					opacity: 0.85
				} )


	geometry = new THREE.ParametricBufferGeometry( THREE.ParametricGeometries[loss](data, lamb, 4,4), 100, 100 );
	object = new THREE.Mesh( geometry, material );
	object.position.set( 0, 0, 0 );
	object.scale.multiplyScalar( 5 );

	scene.add( object );
}

THREE.ParametricGeometries = {

	unregularized: function ( x, lamb, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var z = (x - w2 * w1 * x)**2;

			target.set( w1, w2, z );
		};
	},

	product: function ( x, lamb, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var z = (x - w2 * w1 * x)**2 + lamb * (w2 * w1)**2;

			target.set( w1, w2, z );
		};
	},

	sum: function ( x, lamb, width, height ) {
		return function ( u, v, target ) {

			u -= 0.5;
			v -= 0.5;

			var w1 = u * width;
			var w2 = v * height;
			var z = (x - w2 * w1 * x)**2 + lamb * (w1**2 + w2**2);

			target.set( w1, w2, z );
		};
	}

};

var loss = "unregularized",
	data = 1,
	lamb = 0.5;
addLandscape(loss, data, lamb);


$("input[name=loss]").on("change", function() {
	loss = $(this).val();
	removeLandscape();
	addLandscape(loss, data, lamb);
});

$("#data").on("input", function() {
	data = $(this).val();
	$("#data-val").text(data);
	removeLandscape();
	addLandscape(loss, data, lamb);
});

$("#lamb").on("input", function() {
	lamb = $(this).val();
	$("#lamb-val").text(lamb);
	removeLandscape();
	addLandscape(loss, data, lamb);
});

