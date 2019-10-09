function init() {
    var obj = {};
    // サイズを指定
    const width  = window.innerWidth;
    const height = window.innerHeight;
    //let   vrot   = 0;
    //let   hrot   = 180; // 角度
    let   zoom   = 600;

    // レンダラーを作成
    const renderer = new THREE.WebGLRenderer({
        canvas: document.querySelector('#WebSenseCanvas')
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    // シーンを作成
    const scene    = new THREE.Scene();
    // カメラを作成
    const camera   = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
    const controls = new THREE.OrbitControls( camera, renderer.domElement );
    camera.position.set(0, 0, -zoom);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    controls.update();
    // 基準線を作成
    staticObjects();
    initPose();
    tick();

    function tick() {
        loadContents();
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(tick);
    }

    function loadContents() {
        fetch('/pose.json')
            .then((res)=>{
                if (res.ok){ return res.json(); }
            }).then((json)=>{
                updatePose(json);
            });
    }

    function cleanupObjects(){
        e = queue.shift();
        scene.remove(e['mesh']);
        e['geometry'].dispose();
        e['material'].dispose();
    }

    function initPose(){
        obj['nose'] = {};
        obj['nose']['mat'] = new THREE.MeshBasicMaterial({color: 0xFF00FF});
        obj['nose']['geo'] = new THREE.SphereGeometry(5, 32, 32);
        obj['nose']['mesh'] = new THREE.Mesh( obj['nose']['geo'], obj['nose']['mat']);
        scene.add(obj['nose']['mesh']);
        obj['lear'] = {};
        obj['lear']['mat'] = new THREE.MeshBasicMaterial({color: 0xFFFF00});
        obj['lear']['geo'] = new THREE.SphereGeometry(5, 32, 32);
        obj['lear']['mesh'] = new THREE.Mesh( obj['lear']['geo'], obj['lear']['mat']);
        scene.add(obj['lear']['mesh']);
        obj['rear'] = {};
        obj['rear']['mat'] = new THREE.MeshBasicMaterial({color: 0xFFFF00});
        obj['rear']['geo'] = new THREE.SphereGeometry(5, 32, 32);
        obj['rear']['mesh'] = new THREE.Mesh( obj['rear']['geo'], obj['rear']['mat']);
        scene.add(obj['rear']['mesh']);
        obj['leye'] = {};
        obj['leye']['mat'] = new THREE.MeshBasicMaterial({color: 0xFF0000});
        obj['leye']['geo'] = new THREE.SphereGeometry(5, 32, 32);
        obj['leye']['mesh'] = new THREE.Mesh( obj['leye']['geo'], obj['leye']['mat']);
        scene.add(obj['leye']['mesh']);
        obj['reye'] = {};
        obj['reye']['mat'] = new THREE.MeshBasicMaterial({color: 0xFF0000});
        obj['reye']['geo'] = new THREE.SphereGeometry(5, 32, 32);
        obj['reye']['mesh'] = new THREE.Mesh( obj['reye']['geo'], obj['reye']['mat']);
        scene.add(obj['reye']['mesh']);

        obj['larm'] = {};
        obj['larm']['mat']  = new THREE.LineBasicMaterial( { linewidth: 100, color: 0xff0000 } );
        obj['larm']['geo']  = new THREE.Geometry();
        obj['larm']['geo'].vertices[0] = new THREE.Vector3(0, 0, 0);
        obj['larm']['geo'].vertices[1] = new THREE.Vector3(100, 100, 100);
        obj['larm']['geo'].vertices[2] = new THREE.Vector3(200, 200, 200);
        obj['larm']['geo'].vertices[3] = new THREE.Vector3(300, 300, 300);
        obj['larm']['geo'].vertices[4] = new THREE.Vector3(400, 400, 400);
        obj['larm']['geo'].vertices[5] = new THREE.Vector3(500, 500, 500);
        obj['larm']['line'] = new THREE.Line( obj['larm']['geo'], obj['larm']['mat'] );
        scene.add(obj['larm']['line']);

        obj['rarm'] = {};
        obj['rarm']['mat']  = new THREE.LineBasicMaterial( { linewidth: 100, color: 0xff0000 } );
        obj['rarm']['geo']  = new THREE.Geometry();
        obj['rarm']['geo'].vertices[0] = new THREE.Vector3(0, 0, 0);
        obj['rarm']['geo'].vertices[1] = new THREE.Vector3(100, 100, 100);
        obj['rarm']['geo'].vertices[2] = new THREE.Vector3(200, 200, 200);
        obj['rarm']['geo'].vertices[3] = new THREE.Vector3(300, 300, 300);
        obj['rarm']['geo'].vertices[4] = new THREE.Vector3(400, 400, 400);
        obj['rarm']['geo'].vertices[5] = new THREE.Vector3(500, 500, 500);
        obj['rarm']['line'] = new THREE.Line( obj['rarm']['geo'], obj['rarm']['mat'] );
        scene.add(obj['rarm']['line']);

        obj['shoulder'] = {};
        obj['shoulder']['mat']  = new THREE.LineBasicMaterial( { linewidth: 100, color: 0xffff00 } );
        obj['shoulder']['geo']  = new THREE.Geometry();
        obj['shoulder']['geo'].vertices[0] = new THREE.Vector3(0, 0, 0);
        obj['shoulder']['geo'].vertices[1] = new THREE.Vector3(100, 100, 100);
        obj['shoulder']['line'] = new THREE.Line( obj['shoulder']['geo'], obj['shoulder']['mat'] );
        scene.add(obj['shoulder']['line']);

        obj['hip'] = {};
        obj['hip']['mat']  = new THREE.LineBasicMaterial( { linewidth: 100, color: 0xffff00 } );
        obj['hip']['geo']  = new THREE.Geometry();
        obj['hip']['geo'].vertices[0] = new THREE.Vector3(0, 0, 0);
        obj['hip']['geo'].vertices[1] = new THREE.Vector3(100, 100, 100);
        obj['hip']['line'] = new THREE.Line( obj['hip']['geo'], obj['hip']['mat'] );
        scene.add(obj['hip']['line']);
    }

    function updatePose(pose){
        if (pose[0] == undefined){return;}
        obj['nose']['mesh'].position.set(
            pose[0]['nose']['point']['0'],
            -pose[0]['nose']['point']['1'],
            pose[0]['nose']['point']['2']);
        obj['leye']['mesh'].position.set(
            pose[0]['leftEye']['point']['0'],
            -pose[0]['leftEye']['point']['1'],
            pose[0]['leftEye']['point']['2']);
            obj['reye']['mesh'].position.set(
            pose[0]['rightEye']['point']['0'],
            -pose[0]['rightEye']['point']['1'],
            pose[0]['rightEye']['point']['2']);
        obj['lear']['mesh'].position.set(
            pose[0]['leftEar']['point']['0'],
            -pose[0]['leftEar']['point']['1'],
            pose[0]['leftEar']['point']['2']);
        obj['rear']['mesh'].position.set(
            pose[0]['rightEar']['point']['0'],
            -pose[0]['rightEar']['point']['1'],
            pose[0]['rightEar']['point']['2']);
    
        obj['larm']['geo'].vertices[0].set(
            pose[0]['leftAnkle']['point']['0'],
            -pose[0]['leftAnkle']['point']['1'],
            pose[0]['leftAnkle']['point']['2']);
        obj['larm']['geo'].vertices[1].set(
            pose[0]['leftKnee']['point']['0'],
            -pose[0]['leftKnee']['point']['1'],
            pose[0]['leftKnee']['point']['2']);
        obj['larm']['geo'].vertices[2].set(
            pose[0]['leftHip']['point']['0'],
            -pose[0]['leftHip']['point']['1'],
            pose[0]['leftHip']['point']['2']);
        obj['larm']['geo'].vertices[3].set(
            pose[0]['leftShoulder']['point']['0'],
            -pose[0]['leftShoulder']['point']['1'],
            pose[0]['leftShoulder']['point']['2']);
        obj['larm']['geo'].vertices[4].set(
            pose[0]['leftElbow']['point']['0'],
            -pose[0]['leftElbow']['point']['1'],
            pose[0]['leftElbow']['point']['2']);
        obj['larm']['geo'].vertices[5].set(
            pose[0]['leftWrist']['point']['0'],
            -pose[0]['leftWrist']['point']['1'],
            pose[0]['leftWrist']['point']['2']);
        obj['larm']['geo'].verticesNeedUpdate = true;
        obj['larm']['geo'].elementNeedUpdate = true;
        obj['larm']['geo'].computeFaceNormals(); 

        obj['rarm']['geo'].vertices[0].set(
            pose[0]['rightAnkle']['point']['0'],
            -pose[0]['rightAnkle']['point']['1'],
            pose[0]['rightAnkle']['point']['2']);
        obj['rarm']['geo'].vertices[1].set(
            pose[0]['rightKnee']['point']['0'],
            -pose[0]['rightKnee']['point']['1'],
            pose[0]['rightKnee']['point']['2']);
        obj['rarm']['geo'].vertices[2].set(
            pose[0]['rightHip']['point']['0'],
            -pose[0]['rightHip']['point']['1'],
            pose[0]['rightHip']['point']['2']);
        obj['rarm']['geo'].vertices[3].set(
            pose[0]['rightShoulder']['point']['0'],
            -pose[0]['rightShoulder']['point']['1'],
            pose[0]['rightShoulder']['point']['2']);
        obj['rarm']['geo'].vertices[4].set(
            pose[0]['rightElbow']['point']['0'],
            -pose[0]['rightElbow']['point']['1'],
            pose[0]['rightElbow']['point']['2']);
            obj['rarm']['geo'].vertices[5].set(
                pose[0]['rightWrist']['point']['0'],
                -pose[0]['rightWrist']['point']['1'],
                pose[0]['rightWrist']['point']['2']);
            obj['rarm']['geo'].verticesNeedUpdate = true;
            obj['rarm']['geo'].elementNeedUpdate = true;
            obj['rarm']['geo'].computeFaceNormals();
            
            obj['shoulder']['geo'].vertices[0].set(
                pose[0]['leftShoulder']['point']['0'],
                -pose[0]['leftShoulder']['point']['1'],
                pose[0]['leftShoulder']['point']['2']);
            obj['shoulder']['geo'].vertices[1].set(
                pose[0]['rightShoulder']['point']['0'],
                -pose[0]['rightShoulder']['point']['1'],
                pose[0]['rightShoulder']['point']['2']);
            obj['shoulder']['geo'].verticesNeedUpdate = true;
            obj['shoulder']['geo'].elementNeedUpdate = true;
            obj['shoulder']['geo'].computeFaceNormals();
    
            obj['hip']['geo'].vertices[0].set(
                pose[0]['leftHip']['point']['0'],
                -pose[0]['leftHip']['point']['1'],
                pose[0]['leftHip']['point']['2']);
            obj['hip']['geo'].vertices[1].set(
                pose[0]['rightHip']['point']['0'],
                -pose[0]['rightHip']['point']['1'],
                pose[0]['rightHip']['point']['2']);
            obj['hip']['geo'].verticesNeedUpdate = true;
            obj['hip']['geo'].elementNeedUpdate = true;
            obj['hip']['geo'].computeFaceNormals();
    }

    // X, Y, Z軸線を引く
    function staticObjects(){
        const l_length = 10000;

        var x_mat = new THREE.LineBasicMaterial( { linewidth: 10, color: 0xff0000 } );
        var x_geo = new THREE.Geometry();
        x_geo.vertices.push(new THREE.Vector3(0, 0, 0));
        x_geo.vertices.push(new THREE.Vector3(l_length, 0, 0));
        
        scene.add( new THREE.Line( x_geo, x_mat ) );

        var y_mat = new THREE.LineBasicMaterial( { linewidth: 10, color: 0x0000ff } );
        var y_geo = new THREE.Geometry();
        y_geo.vertices.push(new THREE.Vector3(0, 0, 0));
        y_geo.vertices.push(new THREE.Vector3(0, l_length, 0));
        scene.add( new THREE.Line( y_geo, y_mat ) );

        var z_mat = new THREE.LineBasicMaterial( { linewidth: 10, color: 0x00ff00 } );
        var z_geo = new THREE.Geometry();
        z_geo.vertices.push(new THREE.Vector3(0, 0, 0));
        z_geo.vertices.push(new THREE.Vector3(0, 0, l_length));
        scene.add( new THREE.Line( z_geo, z_mat ) );
    }
}

window.addEventListener('load', init);