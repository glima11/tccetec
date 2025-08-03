from flask import Flask, request, jsonify, send_from_directory, abort, render_template_string
import os
import base64
import re

app = Flask(__name__)

UPLOAD_FOLDER = 'photos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_str, filename):
    # Remove o cabeçalho e decodifica base64
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    img_data = base64.b64decode(base64_data)
    with open(filename, 'wb') as f:
        f.write(img_data)

@app.route('/')
def index():
    # Página SPA com frontend completo
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cadastro SPA com Webcam</title>
  <style>
    body {
      font-family: Arial, sans-serif; 
      max-width: 900px; 
      margin: 20px auto; 
      padding: 0 10px;
    }
    #gallery {
      display: flex; 
      flex-wrap: wrap; 
      gap: 15px; 
      margin-top: 20px;
    }
    .photo-item {
      position: relative;
      width: 150px; 
      text-align: center; 
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 5px;
      box-shadow: 1px 1px 6px #ccc;
    }
    .photo-item img {
      width: 150px; 
      border-radius: 4px;
    }
    .delete-btn {
      position: absolute;
      top: 4px; right: 4px;
      background: rgba(255,0,0,0.8);
      border: none;
      color: white;
      cursor: pointer;
      border-radius: 3px;
      padding: 2px 6px;
      font-size: 12px;
    }
    label {
      font-weight: bold;
    }
    input[type=text], input[type=tel] {
      width: 250px;
      padding: 6px 8px;
      margin: 6px 0 12px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
    }
    button {
      background-color: #0275d8;
      border: none;
      padding: 8px 16px;
      color: white;
      border-radius: 4px;
      font-size: 15px;
      cursor: pointer;
    }
    button:hover {
      background-color: #025aa5;
    }
    video, canvas, img#photo {
      border: 1px solid #ccc;
      border-radius: 6px;
      display: block;
      margin-top: 10px;
    }
    #photo {
      max-width: 320px;
    }
  </style>
</head>
<body>
  <h1>ETEC Jaraguá - TCC grupo Giovanni - Reconhecimento Facial e Liberação de Acesso</h1>

  <form id="form" autocomplete="off">
    <label for="nome">Nome:</label><br>
    <input type="text" id="nome" name="nome" required maxlength="100"><br>

    <label for="telefone">Telefone:</label><br>
    <input type="tel" id="telefone" name="telefone" pattern="[0-9+()\\-\\s]{8,20}" required maxlength="20"><br>

    <label>Vídeo da Webcam:</label>
    <video id="video" width="320" height="240" autoplay></video><br>

    <button type="button" id="capture">Capturar Foto</button>

    <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>

    <img id="photo" alt="Foto Capturada" style="display:none;"><br>

    <button type="submit">Enviar Cadastro</button>
  </form>

  <h2>Galeria de Fotos</h2>
  <div id="gallery">Carregando galeria...</div>

<script>
  // Elementos DOM
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const photo = document.getElementById('photo');
  const captureButton = document.getElementById('capture');
  const form = document.getElementById('form');
  const gallery = document.getElementById('gallery');

  let capturedImage = null;

  // Acessa a webcam e stream para <video>
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
    } catch (err) {
      alert('Erro ao acessar a webcam: ' + err.message);
    }
  }

  startWebcam();

  captureButton.onclick = () => {
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    capturedImage = canvas.toDataURL('image/jpeg');
    photo.src = capturedImage;
    photo.style.display = 'block';
  };

  async function loadGallery() {
    gallery.innerHTML = 'Carregando galeria...';
    try {
      const res = await fetch('/photos');
      if (!res.ok) throw new Error('Falha carregando as fotos');
      const files = await res.json();

      if (files.length === 0) {
        gallery.innerHTML = '<p>Nenhuma foto cadastrada.</p>';
        return;
      }

      gallery.innerHTML = '';
      files.forEach(filename => {
        const div = document.createElement('div');
        div.className = 'photo-item';

        const img = document.createElement('img');
        img.src = `/photos/${filename}`;
        img.alt = filename;

        const btn = document.createElement('button');
        btn.textContent = `Excluir ${filename}`;
        btn.className = 'delete-btn';
        btn.onclick = async () => {
          if (confirm(`Deseja excluir a foto ${filename}?`)) {
            try {
              const delRes = await fetch(`/photos/${filename}`, { method: 'DELETE' });
              if (delRes.ok) {
                alert('Foto excluída com sucesso!');
                loadGallery();
              } else {
                alert('Erro ao excluir foto');
              }
            } catch {
              alert('Erro de rede ao excluir foto');
            }
          }
        };

        div.appendChild(img);
        div.appendChild(btn);
        gallery.appendChild(div);
      });
    } catch (e) {
      gallery.innerHTML = '<p>Erro ao carregar galeria.</p>';
      console.error(e);
    }
  }

  form.onsubmit = async (e) => {
    e.preventDefault();
    if (!capturedImage) {
      alert('Por favor, capture uma foto antes de enviar.');
      return;
    }
    const nome = document.getElementById('nome').value.trim();
    const telefone = document.getElementById('telefone').value.trim();

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nome,
          telefone,
          foto_data: capturedImage
        })
      });
      if (res.ok) {
        alert('Cadastro enviado com sucesso!');
        form.reset();
        photo.style.display = 'none';
        capturedImage = null;
        loadGallery();
      } else {
        const data = await res.json();
        alert('Erro no envio: ' + (data.error || 'Desconhecido'));
      }
    } catch (err) {
      alert('Erro de rede: ' + err.message);
    }
  };

  // Carregar galeria ao iniciar a página
  loadGallery();
</script>
</body>
</html>
    ''')

@app.route('/upload', methods=['POST'])
def upload():
    if not request.is_json:
        return jsonify({'error': 'Conteúdo JSON esperado'}), 400
    data = request.get_json()
    nome = data.get('nome', '').strip()
    telefone = data.get('telefone', '').strip()
    foto_data = data.get('foto_data', '')

    if not nome or not telefone or not foto_data:
        return jsonify({'error': 'Campos nome, telefone e foto são obrigatórios'}), 400

    # Sanitiza o nome para ser usado no nome do arquivo
    safe_nome = "".join(c for c in nome if c.isalnum() or c in (' ', '-', '_')).rstrip()
    if not safe_nome:
        safe_nome = "foto"
    filename = safe_nome + '.jpg'

    save_path = os.path.join(UPLOAD_FOLDER, filename)

    # Se arquivo já existir, adicionar número para não sobrescrever
    if os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while True:
            new_filename = f"{base}_{i}{ext}"
            new_path = os.path.join(UPLOAD_FOLDER, new_filename)
            if not os.path.exists(new_path):
                filename = new_filename
                save_path = new_path
                break
            i += 1

    try:
        save_base64_image(foto_data, save_path)
    except Exception as e:
        return jsonify({'error': 'Erro ao salvar a foto: ' + str(e)}), 500

    # Opcional: salvar nome e telefone em um arquivo texto para registro
    with open(os.path.join(UPLOAD_FOLDER, 'dados.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{nome};{telefone};{filename}\n")

    return jsonify({'message': 'Upload realizado com sucesso', 'filename': filename}), 200

@app.route('/photos', methods=['GET'])
def list_photos():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    return jsonify(files)

@app.route('/photos/<filename>', methods=['GET'])
def serve_photo(filename):
    if not allowed_file(filename):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/photos/<filename>', methods=['DELETE'])
def delete_photo(filename):
    if not allowed_file(filename):
        abort(404)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({'message': 'Foto excluída com sucesso'}), 200
        except Exception as e:
            return jsonify({'error': 'Erro ao excluir foto: ' + str(e)}), 500
    else:
        return jsonify({'error': 'Arquivo não encontrado'}), 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
