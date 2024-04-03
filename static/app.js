/*
    Quadrotor visuals with THREE.js and data from a websocket connection

    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

class quadRotor {
    constructor(width, length, diameter) {
        const xs = [length, length, -length, -length];
        const ys = [-width, width, -width, width];
        const rotorGeo = new THREE.CircleGeometry( .5*diameter, 16 );
        const edges = new THREE.EdgesGeometry( rotorGeo ); 
        const mat = new THREE.LineBasicMaterial( { color: 0x000000 } );

        const triangleGeometry = new THREE.BufferGeometry();
        const vertices = new Float32Array ( [
                    length*1.3, 0, 0,  // Top
                    length*0.7, -width/4, 0, // Bottom left
                    length*0.7, +width/4, 0,   // Bottom right
        ] );
        const indices = [
            0,1,2, // top
            0,2,1, // bottom (right hand rule)
        ];

        triangleGeometry.setIndex( indices );
        triangleGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

        const darkGreenMaterial = new THREE.MeshBasicMaterial({ color: 0x006400 });
        const triangleMesh = new THREE.Mesh(triangleGeometry, darkGreenMaterial);

        this.obj = new THREE.Object3D();
        this.obj.add(triangleMesh);

        var rotorLines = []
        for (let i = 0; i < 4; i++) {
            var line = new THREE.LineSegments(edges, mat);
            this.obj.add(line);
            rotorLines.push(line);
            rotorLines[i].position.x = xs[i];
            rotorLines[i].position.y = ys[i];
        }
    }
    addToScene(theScene) {
        theScene.add(this.obj);
    }
    setPose(pos, quat) {
        this.obj.position.x = pos[0];
        this.obj.position.y = pos[1];
        this.obj.position.z = pos[2];
        this.obj.quaternion.w = quat[0];
        this.obj.quaternion.x = quat[1];
        this.obj.quaternion.y = quat[2];
        this.obj.quaternion.z = quat[3];
    }
}

export function dressUpScene(scene, camera) {
    // scene with white background
    scene.background = new THREE.Color(0xffffff);

    // ground plane grid in the xy-plane
    const gd = new THREE.GridHelper( 10, 10 );
    gd.rotation.x = -0.5*3.1415
    scene.add( gd );

    // coordinate vectors (RGB --> XYZ (north-east-down))
    const xdir = new THREE.Vector3( 1, 0, 0 );
    const ydir = new THREE.Vector3( 0, 1, 0 );
    const zdir = new THREE.Vector3( 0, 0, 1 );
    const origin = new THREE.Vector3( 0, 0, 0, );
    const length = 1.0;
    const xAxis = new THREE.ArrowHelper( xdir, origin, length, 0xff0000, 0.2*length, 0.1*length );
    const yAxis = new THREE.ArrowHelper( ydir, origin, length, 0x00ff00, 0.2*length, 0.1*length );
    const zAxis = new THREE.ArrowHelper( zdir, origin, length, 0x0000ff, 0.2*length, 0.1*length );
    scene.add( xAxis ); scene.add( yAxis ); scene.add( zAxis );

    // camera, such that North East Down makes sense
    camera.up.set( 0, 0, -1 ); // for orbit controls to make sense
    camera.position.x = -4;
    camera.position.y = 3;
    camera.position.z = -3;
    camera.setRotationFromEuler( new THREE.Euler(-110*3.1415/180, 0, 55 * 3.1415/180, 'ZYX'))

}


function updateVisualization(data) {
    if (!idList.includes(data.id)) {
        // never seen this is, add new quadrotor to scene
        idList.push(data.id);
        var newQuadrotor = new quadRotor(0.1, 0.06, 0.07);
        craftList.push(newQuadrotor);
        newQuadrotor.addToScene(scene);
    }
    var idx = idList.indexOf(data.id);
    craftList[idx].setPose(data.pos, data.quat);
}

function animate() {
	requestAnimationFrame( animate );
    controls.update();
	renderer.render( scene, camera );
}

function startWebsocket() {
  socket = new WebSocket('ws://localhost:8765');

  socket.onopen = function(event) {
      console.log('WebSocket connection established.');
  };

  socket.onmessage = function(event) {
      // Update visualization with received data
      const data = JSON.parse(event.data);
      updateVisualization(data);
  };

  socket.onclose = function(){
    // connection closed, discard old websocket and create a new one in 5s
    socket = null;
    //setTimeout(startWebsocket, 200);
  }

  socket.onerror = function(err) {
    //console.error('Socket encountered error: ', err.message, 'Closing socket');
    socket.close();
  };
}

function retryConnection() {
    if ((socket === null) || (socket.readyState != 1)) {
        console.log('(Re)trying WebSocket connection...');
        startWebsocket();
    }
    setTimeout(retryConnection, 500);
}


var socket = null;
var idList = []
var craftList = []
//startWebsocket();
retryConnection();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 0.1, 1000 );

dressUpScene(scene, camera);

// webGL renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement )

const controls = new OrbitControls( camera, renderer.domElement );

animate();

document.addEventListener('keydown', function(event) {
    // Check if the space bar is pressed
    if (event.code === 'Space') {
        // Reset OrbitControls state
        controls.reset();
    }
});