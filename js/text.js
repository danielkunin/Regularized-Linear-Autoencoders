// https://github.com/stemkoski/stemkoski.github.com/blob/master/Three.js/Sprite-Text-Labels.html
function makeTextSprite( message, parameters )
{
	if ( parameters === undefined ) parameters = {};
	
	var fontface = parameters.hasOwnProperty("fontface") ? 
		parameters["fontface"] : "sans serif";
	
	var fontsize = parameters.hasOwnProperty("fontsize") ? 
		parameters["fontsize"] : 18;
		
	var canvas = document.createElement('canvas');
	canvas.width  = 256;
	canvas.height = 128;
	var context = canvas.getContext('2d');
	context.font = fontsize + "px " + fontface;
    
	// get size data (height depends only on font size)
	var metrics = context.measureText( message );
	var textWidth = metrics.width;
	
	// text color
	context.fillStyle = "rgba(0, 0, 0, 1.0)";
	context.fillText( message, (canvas.width - textWidth) / 2, canvas.height / 2 + fontsize / 2);
	
	// canvas contents will be used for a texture
	var texture = new THREE.Texture(canvas) 
	texture.needsUpdate = true;
	var spriteMaterial = new THREE.SpriteMaterial( { map: texture} );
	var sprite = new THREE.Sprite( spriteMaterial );
	sprite.scale.set(10,5,1.0);
	return sprite;	
}