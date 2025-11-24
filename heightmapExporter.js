/**
 * Heightmap Exporter - Enhanced export for Unreal Engine 5.6
 */
class HeightmapExporter {
  /**
   * Generate 16-bit grayscale heightmap from terrain data
   */
  static generateHeightmapData(heightMap, exportSize) {
    const sourceSize = heightMap.length;
    const heightData = new Uint16Array(exportSize * exportSize);
    
    // Find min/max for normalization
    let minHeight = Infinity;
    let maxHeight = -Infinity;
    for (let x = 0; x < sourceSize; x++) {
      for (let y = 0; y < sourceSize; y++) {
        const h = heightMap[x][y];
        if (h < minHeight) minHeight = h;
        if (h > maxHeight) maxHeight = h;
      }
    }
    
    const heightRange = maxHeight - minHeight || 1;
    
    for (let y = 0; y < exportSize; y++) {
      for (let x = 0; x < exportSize; x++) {
        const srcX = Math.floor((x / exportSize) * (sourceSize - 1));
        const srcY = Math.floor((y / exportSize) * (sourceSize - 1));
        
        const fracX = (x / exportSize) * (sourceSize - 1) - srcX;
        const fracY = (y / exportSize) * (sourceSize - 1) - srcY;
        
        const x0 = Math.min(srcX, sourceSize - 1);
        const y0 = Math.min(srcY, sourceSize - 1);
        const x1 = Math.min(srcX + 1, sourceSize - 1);
        const y1 = Math.min(srcY + 1, sourceSize - 1);
        
        const h00 = heightMap[x0][y0] || 0;
        const h10 = heightMap[x1][y0] || 0;
        const h01 = heightMap[x0][y1] || 0;
        const h11 = heightMap[x1][y1] || 0;
        
        const h0 = h00 * (1 - fracX) + h10 * fracX;
        const h1 = h01 * (1 - fracX) + h11 * fracX;
        const heightValue = h0 * (1 - fracY) + h1 * fracY;
        
        const normalized = (heightValue - minHeight) / heightRange;
        const value16bit = Math.max(0, Math.min(65535, normalized * 65535));
        
        const pixelIndex = y * exportSize + x;
        heightData[pixelIndex] = Math.round(value16bit);
      }
    }
    
    return heightData;
  }

  /**
   * Create proper 16-bit PNG with deflate compression
   */
  static createProper16BitPNG(heightData, exportSize) {
    const pngSignature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    
    // CRC32 calculation
    const crc32 = (function() {
      const table = new Uint32Array(256);
      for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
          c = ((c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1));
        }
        table[n] = c >>> 0;
      }
      return function(data) {
        let crc = 0 ^ (-1);
        for (let i = 0; i < data.length; i++) {
          crc = (crc >>> 8) ^ table[(crc ^ data[i]) & 0xFF];
        }
        return (crc ^ (-1)) >>> 0;
      };
    })();

    const writeBE32 = (arr, offset, value) => {
      arr[offset] = (value >>> 24) & 0xFF;
      arr[offset + 1] = (value >>> 16) & 0xFF;
      arr[offset + 2] = (value >>> 8) & 0xFF;
      arr[offset + 3] = value & 0xFF;
    };

    // IHDR chunk
    const ihdr = new Uint8Array(25);
    writeBE32(ihdr, 0, 13);
    ihdr[4] = 'I'.charCodeAt(0);
    ihdr[5] = 'H'.charCodeAt(0);
    ihdr[6] = 'D'.charCodeAt(0);
    ihdr[7] = 'R'.charCodeAt(0);
    writeBE32(ihdr, 8, exportSize);
    writeBE32(ihdr, 12, exportSize);
    ihdr[16] = 16; // 16-bit depth
    ihdr[17] = 0; // Grayscale
    ihdr[18] = 0; // Deflate compression
    ihdr[19] = 0; // Adaptive filtering
    ihdr[20] = 0; // No interlace
    writeBE32(ihdr, 21, crc32(ihdr.slice(4, 21)));

    // Create scanline data
    const rawData = new Uint8Array(exportSize * exportSize * 2 + exportSize);
    let rawIndex = 0;

    for (let y = 0; y < exportSize; y++) {
      rawData[rawIndex++] = 0; // Filter type none
      for (let x = 0; x < exportSize; x++) {
        const value = heightData[y * exportSize + x];
        rawData[rawIndex++] = (value >>> 8) & 0xFF; // High byte
        rawData[rawIndex++] = value & 0xFF; // Low byte
      }
    }

    // Compress with pako if available
    let compressedData;
    if (typeof pako !== 'undefined') {
      compressedData = pako.deflate(rawData, { level: 9 });
    } else {
      console.error('Pako library not loaded. Using uncompressed data.');
      compressedData = rawData;
    }

    // IDAT chunk
    const idatLength = compressedData.length;
    const idat = new Uint8Array(idatLength + 12);
    writeBE32(idat, 0, idatLength);
    idat[4] = 'I'.charCodeAt(0);
    idat[5] = 'D'.charCodeAt(0);
    idat[6] = 'A'.charCodeAt(0);
    idat[7] = 'T'.charCodeAt(0);
    idat.set(compressedData, 8);
    writeBE32(idat, 8 + idatLength, crc32(idat.slice(4, 8 + idatLength)));

    // IEND chunk
    const iend = new Uint8Array(12);
    writeBE32(iend, 0, 0);
    iend[4] = 'I'.charCodeAt(0);
    iend[5] = 'E'.charCodeAt(0);
    iend[6] = 'N'.charCodeAt(0);
    iend[7] = 'D'.charCodeAt(0);
    writeBE32(iend, 8, 0xae426082);

    // Combine all chunks
    const totalLength = pngSignature.length + ihdr.length + idat.length + iend.length;
    const pngData = new Uint8Array(totalLength);
    let offset = 0;
    pngData.set(pngSignature, offset);
    offset += pngSignature.length;
    pngData.set(ihdr, offset);
    offset += ihdr.length;
    pngData.set(idat, offset);
    offset += idat.length;
    pngData.set(iend, offset);

    return new Blob([pngData], { type: 'image/png' });
  }

  /**
   * Export as OBJ mesh format
   */
  static exportOBJ(heightMap, heightScale) {
    try {
      const size = heightMap.length;
      let objData = '# Terrain Mesh\n';
      objData += `# Generated from terrain with ${size}x${size} resolution\n`;
      objData += `# Scale: ${heightScale}x\n\n`;

      const spacing = 100 / size;
      let vertexCount = 0;

      // Write vertices
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const posX = (x - size / 2) * spacing;
          const posY = heightMap[x][y] * heightScale;
          const posZ = (y - size / 2) * spacing;
          objData += `v ${posX.toFixed(2)} ${posY.toFixed(2)} ${posZ.toFixed(2)}\n`;
          vertexCount++;
        }
      }

      // Write faces
      objData += '\n# Faces\n';
      for (let y = 0; y < size - 1; y++) {
        for (let x = 0; x < size - 1; x++) {
          const v1 = y * size + x + 1;
          const v2 = y * size + (x + 1) + 1;
          const v3 = (y + 1) * size + x + 1;
          const v4 = (y + 1) * size + (x + 1) + 1;

          objData += `f ${v1} ${v2} ${v3}\n`;
          objData += `f ${v2} ${v4} ${v3}\n`;
        }
      }

      const blob = new Blob([objData], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `terrain_${size}x${size}.obj`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      alert(`✓ OBJ exported: ${size}x${size} mesh with ${vertexCount} vertices`);
    } catch (error) {
      alert('Error exporting OBJ: ' + error.message);
    }
  }

  /**
   * Export as JSON data
   */
  static exportJSON(heightMap, params) {
    try {
      const jsonData = {
        version: '1.0',
        type: 'terrain_heightmap',
        timestamp: new Date().toISOString(),
        metadata: {
          resolution: heightMap.length,
          heightScale: params.heightScale || 20,
          roughness: params.roughness || 0.8,
          frequency: params.frequency || 1.0,
          amplitude: params.amplitude || 1.0,
          noiseType: params.noiseType || 'simplex'
        },
        heightmap: heightMap
      };

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `terrain_preset_${new Date().getTime()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      alert(`✓ JSON exported: ${heightMap.length}x${heightMap.length} heightmap data`);
    } catch (error) {
      alert('Error exporting JSON: ' + error.message);
    }
  }

  /**
   * Save terrain preset to localStorage
   */
  static savePreset(heightMap, params, name = null) {
    try {
      const presetName = name || prompt('Enter preset name:', `Terrain_${new Date().getTime()}`);
      if (!presetName) return;

      const preset = {
        name: presetName,
        timestamp: new Date().toISOString(),
        params: params,
        heightmapSize: heightMap.length
      };

      const presets = JSON.parse(localStorage.getItem('terrainPresets') || '{}');
      presets[presetName] = preset;
      localStorage.setItem('terrainPresets', JSON.stringify(presets));

      alert(`✓ Preset saved: "${presetName}"\n\nReload the page to load this preset.`);
    } catch (error) {
      alert('Error saving preset: ' + error.message);
    }
  }

  /**
   * Load preset from localStorage
   */
  static loadPresets() {
    try {
      const presets = JSON.parse(localStorage.getItem('terrainPresets') || '{}');
      return presets;
    } catch (error) {
      console.error('Error loading presets:', error);
      return {};
    }
  }

  /**
   * Create preview canvas
   */
  static createPreviewCanvas(heightData, exportSize) {
    const canvas = document.createElement('canvas');
    canvas.width = exportSize;
    canvas.height = exportSize;
    const ctx = canvas.getContext('2d');
    
    const imageData = ctx.createImageData(exportSize, exportSize);
    const data = imageData.data;
    
    for (let i = 0; i < heightData.length; i++) {
      const value8bit = Math.floor(heightData[i] / 256);
      data[i * 4] = value8bit;
      data[i * 4 + 1] = value8bit;
      data[i * 4 + 2] = value8bit;
      data[i * 4 + 3] = 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  /**
   * Create preview UI
   */
  static createPreview(previewCanvas, exportSize, heightData, pngBlob) {
    const container = document.createElement('div');
    container.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.95);
      border: 2px solid #00FF00;
      padding: 20px;
      z-index: 10000;
      max-width: 90vw;
      max-height: 90vh;
      overflow: auto;
      border-radius: 4px;
      font-family: "Courier New", monospace;
    `;
    
    const title = document.createElement('h3');
    title.textContent = '✓ 16-bit Grayscale Heightmap (UE5.6 Compatible)';
    title.style.color = '#00FF00';
    title.style.marginBottom = '10px';
    title.style.marginTop = '0';
    
    const displayCanvas = document.createElement('canvas');
    displayCanvas.width = Math.min(400, previewCanvas.width);
    displayCanvas.height = Math.min(400, previewCanvas.height);
    displayCanvas.style.cssText = 'border: 1px solid #00FF00; display: block; margin: 15px 0; image-rendering: pixelated;';
    
    const dctx = displayCanvas.getContext('2d');
    dctx.drawImage(previewCanvas, 0, 0, displayCanvas.width, displayCanvas.height);
    
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < heightData.length; i++) {
      minVal = Math.min(minVal, heightData[i]);
      maxVal = Math.max(maxVal, heightData[i]);
    }
    
    const stats = document.createElement('div');
    stats.style.cssText = 'background: rgba(0,255,0,0.1); padding: 10px; margin: 10px 0; border-left: 2px solid #00FF00;';
    stats.innerHTML = `
      <strong style="color: #00FF00;">Heightmap Statistics:</strong><br>
      <span style="color: #CCCCCC;">Resolution: ${exportSize}x${exportSize}px</span><br>
      <span style="color: #CCCCCC;">Format: 16-bit Grayscale PNG (Compressed)</span><br>
      <span style="color: #CCCCCC;">Min Height: ${minVal} | Max Height: ${maxVal}</span><br>
      <span style="color: #CCCCCC;">File Size: ${(pngBlob.size / 1024).toFixed(2)} KB</span>
    `;
    
    const instructions = document.createElement('div');
    instructions.style.cssText = 'background: rgba(0,255,0,0.05); padding: 10px; margin: 10px 0; border: 1px solid #00FF00; border-radius: 2px;';
    instructions.innerHTML = `
      <strong style="color: #00FF00;">UE 5.6 Import Steps:</strong><br>
      <span style="color: #CCCCCC;">
        1. In UE5.6, create a new Landscape<br>
        2. Click "New" or "Import from File"<br>
        3. Select this PNG file<br>
        4. Set section size to 63×63<br>
        5. Click Create<br>
        <strong style="color: #00FF00;">✓ Heightmap should now import successfully!</strong>
      </span>
    `;
    
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = '⬇ Download PNG';
    downloadBtn.style.cssText = `
      background: #00FF00;
      color: #000;
      border: none;
      padding: 10px 20px;
      cursor: pointer;
      border-radius: 2px;
      font-weight: bold;
      margin-top: 10px;
      margin-right: 10px;
      font-size: 1rem;
    `;
    downloadBtn.onclick = () => {
      const url = URL.createObjectURL(pngBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `heightmap_${exportSize}x${exportSize}_16bit.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    };
    downloadBtn.onmouseover = () => downloadBtn.style.background = '#FF6600';
    downloadBtn.onmouseout = () => downloadBtn.style.background = '#00FF00';
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.style.cssText = `
      background: #00FF00;
      color: #000;
      border: none;
      padding: 10px 20px;
      cursor: pointer;
      border-radius: 2px;
      font-weight: bold;
      margin-top: 10px;
    `;
    closeBtn.onclick = () => container.remove();
    closeBtn.onmouseover = () => closeBtn.style.background = '#FF6600';
    closeBtn.onmouseout = () => closeBtn.style.background = '#00FF00';
    
    const buttonContainer = document.createElement('div');
    buttonContainer.appendChild(downloadBtn);
    buttonContainer.appendChild(closeBtn);
    
    container.appendChild(title);
    container.appendChild(displayCanvas);
    container.appendChild(stats);
    container.appendChild(instructions);
    container.appendChild(buttonContainer);
    
    return container;
  }

  /**
   * Main export function
   */
  static async exportHeightmap(heightMap, exportSize, heightScale) {
    try {
      if (!heightMap || heightMap.length === 0) {
        alert('Error: No heightmap data available. Generate terrain first!');
        return;
      }
      
      const heightData = this.generateHeightmapData(heightMap, exportSize);
      const pngBlob = this.createProper16BitPNG(heightData, exportSize);
      const previewCanvas = this.createPreviewCanvas(heightData, exportSize);
      const preview = this.createPreview(previewCanvas, exportSize, heightData, pngBlob);
      document.body.appendChild(preview);
      
    } catch (error) {
      console.error('Heightmap export error:', error);
      alert('Error exporting heightmap: ' + error.message);
    }
  }
}

// Export for use as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HeightmapExporter;
}
