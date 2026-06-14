const brightColors = ["#931F1D", "#5398be", "#b6cca1", "#8764A6", "#9B5752", "#4259B3", "#6DAD94", "#E3AF36"];

let canvas = null;
let ctx = null;

let FFTLength = 256;
let Radix2Step = 0;
let Radix2StepCount = 1;

let GridWidth = 1;
let GridHeight = 1;

let mousePxX = -100;
let mousePxY = -100;

let containerWidth = 0;
let containerHeight = 0;

let SIMDSize = 8;

const DisplayType = Object.freeze({
  AngleIndices: "Angles",
  ElementIndices: "Elements",
  None: "None",
});

const BitFlipMode = Object.freeze({
  Standard: "Standard",
  Reversed: "Reversed",
  Progressive: "Progressive",
  MiddleOut: "MiddleOut",
});

let display = DisplayType.ElementIndices;
let bitFlipMode = BitFlipMode.MiddleOut;
let radix = 2;


function frac(v)
  { return v - Math.trunc(v); }



function signNoZero(v)
  { return (v == 0) ? 1 : Math.sign(v); }



function normalize(x, y)
{
  const len = Math.sqrt(x * x + y * y);
  return [x / len, y / len];
}



function dot(ax, ay, bx, by)
  { return (ax * bx + ay * by); }



function wrap(v, lim)
  { return (v % lim + lim) % lim; }



function DrawGrid()
{
  const width = canvas.width;
  const height = canvas.height;

  ctx.beginPath();
  ctx.lineWidth = 1;
  ctx.strokeStyle = (FFTLength < 1024) ? "#afafaf" : "#6f6f6f";

  for (let y = 0; y < GridHeight; y++)
  {
    const yPx = Math.round(y * height / GridHeight);
    ctx.moveTo(0, yPx);
    ctx.lineTo(width, yPx);
  }

  for (let x = 0; x < GridWidth; x++)
  {
    const xPx = Math.round(x * width / GridWidth);
    ctx.moveTo(xPx, 0);
    ctx.lineTo(xPx, height);
  }

  ctx.stroke();
  ctx.closePath();
}

function bitCountFromCount(count)
{
  return Math.round(Math.log2(count));
}

function bitReverse(i, bitCount = bitCountFromCount(FFTLength / 2))
{
  i = ((i >>> 1) & 0x55555555) | ((i & 0x55555555) << 1);
  i = ((i >>> 2) & 0x33333333) | ((i & 0x33333333) << 2);
  i = ((i >>> 4) & 0x0f0f0f0f) | ((i & 0x0f0f0f0f) << 4);
  i = ((i >>> 8) & 0x00ff00ff) | ((i & 0x00ff00ff) << 8);
  i = ((i >>> 16) & 0x0000ffff) | ((i & 0x0000ffff) << 16);
  return i >>> (32 - bitCount);
}

function partialBitReverse(i, mode = bitFlipMode, radix2Step = (Radix2Step), bitCount = bitCountFromCount(FFTLength / 2))
{
  if (mode == BitFlipMode.Standard)
    { return i; }
  const ri = bitReverse(i, bitCount);
  if (mode == BitFlipMode.Reversed)
    { return ri; }

  if (mode == BitFlipMode.Progressive)
  {
    let mask = (FFTLength - 1) >>> 1;

    mask >>>= 2 * radix2Step;
    mask <<= radix2Step;

    return ((i & mask) | (ri & ~mask));
  }
  else
  {
    let mask = (FFTLength - 1) >>> 1;

    let reverseCount = (bitCount - radix2Step);
    /*// if (reverseCount > 0) Need to do a fixup at end
    {
      reverseCount++;
    }*/
    mask >>>= 2 * reverseCount;
    mask <<= reverseCount;

    return ((ri & mask) | (i & ~mask));
  }
}

function orderedIndexFromAbsoluteIndex(i, mode = undefined, radix2Step = undefined)
{
  if (mode === undefined)
  {
    mode = bitFlipMode;
  }
  if (radix2Step === undefined)
  {
    radix2Step = Radix2Step;
  }
  return (partialBitReverse(Math.floor(i / 2), mode, radix2Step) * 2) | (i & 1);
}


function DrawBox(xu, yu, w, h, index, color, textColor)
{
  const x = xu * w;
  const y = yu * h;
  ctx.fillStyle = color;
  ctx.fillRect(x,y, w, h);
  ctx.closePath();
  ctx.beginPath();
  ctx.fillStyle = textColor;
  ctx.fillText(`${index}`, x + w / 2, y + h / 2);
  ctx.closePath();
}

function DrawRadix2(reverseSpaceGridIndex, subFFTLength, displayWidth, displayHeight, simdIndex)
{
  const subFFTIndex = Math.floor(reverseSpaceGridIndex / subFFTLength);
  const innerIndex = reverseSpaceGridIndex - subFFTIndex * subFFTLength;
  const innerIndexMasked = innerIndex >> 1;

  let r2Real0 = 0;
  let r2Imag0 = 0;
  let r2Real1 = 0;
  let r2Imag1 = 0;
  let realValue = false;
  if (innerIndexMasked == 0 || innerIndexMasked == subFFTLength / 4)
  {
    realValue = true;
    // This one mixes with this sub-FFT's start
    r2Real0 = r2Imag0 = subFFTIndex * subFFTLength + (innerIndex & 1);
    r2Real1 = r2Imag1 = r2Real0 + subFFTLength / 2;
    r2Imag0 = r2Real0 + subFFTLength;
    r2Imag1 = r2Real1 + subFFTLength;
  }
  else if (innerIndexMasked < subFFTLength / 4)
  {
    // This one has a corresponding imaginary component at subFFTLength / 2 - innerIndexMasked
    // and mixes with the next sub-FFT's start (assuming the sub FFT index is 0)
    const subFFTIndex0 = (subFFTIndex & 0xfffffffe);
    const imagMasked = subFFTLength / 2 - innerIndexMasked;
    r2Real0 = subFFTIndex0 * subFFTLength + innerIndex;
    r2Imag0 = subFFTIndex0 * subFFTLength + (imagMasked << 1) | (innerIndex & 1);
    r2Real1 = r2Real0 + subFFTLength;
    r2Imag1 = r2Imag0 + subFFTLength;
  }
  else
  {
    // This one has a corresponding real component at subFFTLength / 2 - innerIndexMasked
    // and mixes with the next sub-FFT's start (assuming the sub FFT index is 0)
    ctx.fillStyle = "#ffafaf"
    const subFFTIndex0 = (subFFTIndex & 0xfffffffe);
    const imagMasked = subFFTLength / 2 - innerIndexMasked;
    r2Imag0 = subFFTIndex0 * subFFTLength + innerIndex;
    r2Real0 = subFFTIndex0 * subFFTLength + (imagMasked << 1) | (innerIndex & 1);
    r2Real1 = r2Real0 + subFFTLength;
    r2Imag1 = r2Imag0 + subFFTLength;
  }

  // Convert the values out of bit reversed space back into whatever our real space is
  r2Real0 = (bitReverse(Math.floor(r2Real0 / 2)) * 2) | (r2Real0 & 1);
  r2Imag0 = (bitReverse(Math.floor(r2Imag0 / 2)) * 2) | (r2Imag0 & 1);
  r2Real1 = (bitReverse(Math.floor(r2Real1 / 2)) * 2) | (r2Real1 & 1);
  r2Imag1 = (bitReverse(Math.floor(r2Imag1 / 2)) * 2) | (r2Imag1 & 1);

  r2Real0 = orderedIndexFromAbsoluteIndex(r2Real0);
  r2Imag0 = orderedIndexFromAbsoluteIndex(r2Imag0);
  r2Real1 = orderedIndexFromAbsoluteIndex(r2Real1);
  r2Imag1 = orderedIndexFromAbsoluteIndex(r2Imag1);

  const r2r0x = r2Real0 % GridWidth;
  const r2r0y = Math.floor(r2Real0 / GridWidth);
  const r2i0x = r2Imag0 % GridWidth;
  const r2i0y = Math.floor(r2Imag0 / GridWidth);
  const r2r1x = r2Real1 % GridWidth;
  const r2r1y = Math.floor(r2Real1 / GridWidth);
  const r2i1x = r2Imag1 % GridWidth;
  const r2i1y = Math.floor(r2Imag1 / GridWidth);

  if (realValue)
  {
    DrawBox(r2r0x, r2r0y, displayWidth, displayHeight, simdIndex, "#afafff", "#000000");
    DrawBox(r2r1x, r2r1y, displayWidth, displayHeight, simdIndex, "#afffff", "#000000");

    DrawBox(r2i0x, r2i0y, displayWidth, displayHeight, simdIndex, "#7e7eee", "#000000");
    DrawBox(r2i1x, r2i1y, displayWidth, displayHeight, simdIndex, "#7eeeee", "#000000");
  }
  else
  {
    DrawBox(r2r0x, r2r0y, displayWidth, displayHeight, simdIndex, "#afffaf", "#000000");
    DrawBox(r2i0x, r2i0y, displayWidth, displayHeight, simdIndex, "#ffafaf", "#000000");
    DrawBox(r2r1x, r2r1y, displayWidth, displayHeight, simdIndex, "#7fff7f", "#000000");
    DrawBox(r2i1x, r2i1y, displayWidth, displayHeight, simdIndex, "#ff7f7f", "#000000");
  }
}

function GetRelatedIndices(orderedGridIndex, radix2Step)
{
  const groupCount = 1 << radix2Step;
  const complexCountPerGroup = Math.floor(FFTLength  / groupCount / 2);
  const halfGroupIndex = Math.floor(orderedGridIndex / complexCountPerGroup);

  if (halfGroupIndex == 0)
  {
    // This is in the first half of the real group so it is correlated to one point
    return [orderedGridIndex, orderedGridIndex + complexCountPerGroup];
  }
  else if (halfGroupIndex == 1)
  {
    // second half of the real group so correlates to 2 points
    return [orderedGridIndex, orderedGridIndex - complexCountPerGroup];
  }
  else
  {
    const groupStartIndexA = Math.floor(orderedGridIndex / complexCountPerGroup) * complexCountPerGroup;
    let t = bitReverse(groupStartIndexA / 2);
    const  subFFTLength = 2 * groupCount;
    const groupStartIndexB = bitReverse(subFFTLength - t) * 2;
    const groupRelativeIndex = orderedGridIndex - groupStartIndexA;

    const groupRealStartIndex = ((halfGroupIndex & 1) == 0) ? groupStartIndexA : groupStartIndexB;
    const groupImagStartIndex = ((halfGroupIndex & 1) == 0) ? groupStartIndexB : groupStartIndexA;
    const secondDelta = (groupRelativeIndex >= (complexCountPerGroup / 2))
      ? -complexCountPerGroup / 2
      : complexCountPerGroup / 2;

    return [
      groupRealStartIndex + groupRelativeIndex,
      groupImagStartIndex + groupRelativeIndex,
      groupRealStartIndex + secondDelta + groupRelativeIndex,
      groupImagStartIndex + secondDelta + groupRelativeIndex,
    ];
  }
}

function RenderGridDisplay()
{
  canvas = document.getElementById("gridDisplay");

  ctx = canvas.getContext("2d");

  ctx.fillStyle = "black";
  const width = canvas.width;
  const height = canvas.height;
  ctx.fillRect(0, 0, width, height);


  let i = 0;
  const realDivisor = (1 << Radix2Step);
  const realCount = Math.max(2, FFTLength / realDivisor);

  const displayWidth = width / GridWidth;
  const displayHeight = height / GridHeight;
  while (i < FFTLength)
  {
    const ri = orderedIndexFromAbsoluteIndex(i);
    const x = (ri % GridWidth) * displayWidth;
    const y = Math.floor(ri / GridWidth) * displayWidth;
    ctx.beginPath();
    if (i < realCount)
    {
      ctx.fillStyle = "#1f1f4f";
    }
    else
    {
      // I forget why this was a thing?
      /*ctx.fillStyle = ((i % realCount) < realCount / 2)
        ? "#1f4f1f"
        : ((i % (realCount * 2)) < realCount) ? "#4f1f1f" : "#4f1f4f";*/
      ctx.fillStyle = ((i % realCount) < realCount / 2)
        ? "#1f4f1f"
        : "#4f1f4f"
    }
    ctx.fillRect(x, y, displayWidth, displayHeight);
    ctx.closePath();

    if (i < realCount * 2)
    {
      ctx.fillStyle = "#1f1f4f";
      ctx.fillRect(x, y + (displayHeight - displayHeight / 8) / 2, displayWidth, displayHeight / 8);
      ctx.closePath();
    }

    const areIndicesVisible = FFTLength < 4096 && display != DisplayType.None;
    if (areIndicesVisible)
    {
      const tx = x + displayWidth / 2;
      const ty = y + displayHeight / 2;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `${Math.floor(displayHeight * 0.4)}px sans-serif`;

      if (display == DisplayType.ElementIndices)
      {
        ctx.beginPath();
        ctx.globalAlpha = (i & 1) == 0 ? 1.0 : 0.08;

        ctx.fillStyle = "#efefef";
        ctx.fillText(`${i}`, tx, ty);
        ctx.closePath();
        ctx.globalAlpha = 1.0;
      }
      else if (realCount > 2)
      {
        ctx.beginPath();
        const wCount = 1 << Radix2Step;
        const w = bitReverse(Math.floor(i / (FFTLength / wCount)), bitCountFromCount(FFTLength / 4));
        ctx.fillStyle = "#efef4f";
        ctx.fillText(`${w}`, tx, ty);
        ctx.closePath();
      }
    }

    i++;
  }

  DrawGrid();

  if (mousePxX >= 0 && mousePxY >= 0 && Radix2Step < Radix2StepCount )
  {
    // There are (and this is easiest to see in bit reversed space) a bunch of sub-FFTs which are all real-input FFTs
    const subFFTLength = (1 << Radix2Step) * 4;
    /*if (Radix2Step >= 2 || bitFlipMode == BitFlipMode.MiddleOut)
    {
      const absoluteGridIndex =
        Math.floor(
          (Math.floor(mousePxY / displayHeight) * GridWidth + Math.floor(mousePxX / displayWidth)) / SIMDSize)
        * SIMDSize;

      for (let i = SIMDSize - 1; i >= 0; i--)
      {
        const orderedGridIndex = orderedIndexFromAbsoluteIndex(absoluteGridIndex + i);
        const reverseSpaceGridIndex = orderedIndexFromAbsoluteIndex(orderedGridIndex, BitFlipMode.Reversed);

        DrawRadix2(reverseSpaceGridIndex, subFFTLength, displayWidth, displayHeight, i);
      }
    }
    else
    {
      const absoluteGridIndex = Math.floor(mousePxY / displayHeight) * GridWidth + Math.floor(mousePxX / displayWidth);
      const orderedGridIndex = orderedIndexFromAbsoluteIndex(absoluteGridIndex);
      const reverseSpaceGridIndex = orderedIndexFromAbsoluteIndex(orderedGridIndex, BitFlipMode.Reversed);
      DrawRadix2(reverseSpaceGridIndex, subFFTLength, displayWidth, displayHeight, "");
    }*/

    const absoluteGridIndex =
      Math.floor((Math.floor(mousePxY / displayHeight) * GridWidth + Math.floor(mousePxX / displayWidth)) / SIMDSize) * SIMDSize;

    //console.log(`Finding things related to ${orderedGridIndex}`);
    let relatedStack = [];
    for (let i = 0; i < SIMDSize; i++)
    {
      const orderedGridIndex = orderedIndexFromAbsoluteIndex(absoluteGridIndex + i);
      relatedStack.push(orderedGridIndex);
    }

    let relatedSet = new Set();
    while (relatedStack.length > 0)
    {
      const r = relatedStack.pop();
      if (relatedSet.has(r))
      {
        continue;
      }

      // console.log(`  Testing ${r}`);
      relatedSet.add(r);

      // First thing's first: the bit-flipped version of this
      const absolute = orderedIndexFromAbsoluteIndex(r, bitFlipMode, Radix2Step);
      const flipped = orderedIndexFromAbsoluteIndex(
        absolute,
        bitFlipMode,
        //(Radix2Step == 0 && bitFlipMode === BitFlipMode.MiddleOut)
          //? 3
          //:
          (Radix2Step + Math.min(Radix2StepCount - Radix2Step, 2)));
      //relatedStack.push(flipped);
      //console.log(`    Flipped: ${flipped}`);

      const relatedCurrentR2 = GetRelatedIndices(r, Radix2Step);
      for (const i of relatedCurrentR2)
      {
        //console.log(`    Related current: ${i}`);
        relatedStack.push(i);
      }
      if (Radix2Step < Radix2StepCount - 1)
      {
        for (const i of GetRelatedIndices(r, Radix2Step + 1))
        {
          //console.log(`    Related next: ${i}`);
          //relatedStack.push(i);
        }
      }
      /*if (Radix2Step == 0 && (bitFlipMode === BitFlipMode.MiddleOut || bitFlipMode == BitFlipMode.Standard))
      {
        for (const i of GetRelatedIndices(r, Radix2Step + 2))
        {
          //console.log(`    Related next: ${i}`);
          relatedStack.push(i);
        }
      }*/
    }

    for (const i of relatedSet)
    {
      const a = orderedIndexFromAbsoluteIndex(i);
      DrawBox(
        a % GridWidth,
        Math.floor(a / GridWidth),
        displayWidth,
        displayHeight,
        i,
        "#efefef",
        "#000000");
    }
  }
}



function UpdateSizeText()
{
  Radix2StepCount = Math.max(1, (Math.log2(FFTLength) - 1));
  if (radix === 4 && Radix2Step < Radix2StepCount)
  {
    // Unless we're at the step count, for radix 2 we need to round to every other step
    Radix2Step -= Radix2Step & 1;
  }
  Radix2Step = Math.min(Radix2Step, Radix2StepCount);

  stepSlider = document.getElementById("stepSlider")
  stepSlider.max = Math.ceil(Radix2StepCount * 2 / radix);
  stepSlider.value = Math.ceil(Radix2Step * 2 / radix);

  document.getElementById("widthText").innerHTML = `FFT Length: ${FFTLength}`;
  document.getElementById("stepText").innerHTML = `Step: ${(Radix2Step == Radix2StepCount) ? "end" : (Radix2Step * 2 / radix)} / ${Math.ceil(Radix2StepCount * 2 / radix)}`;
  document.getElementById("bitFlipToggle").textContent = `Ordering: ${bitFlipMode}`
  document.getElementById("showIndicesToggle").textContent = `Indices: ${display}`
  document.getElementById("radixToggle").textContent = `Radix: ${radix}`
  document.getElementById("simdToggle").textContent = `SIMD Size: ${SIMDSize}`;

  GridHeight = 1 << Math.floor(Math.log2(Math.sqrt(FFTLength)));
  GridWidth = FFTLength / GridHeight;

  canvas = document.getElementById("gridDisplay");
  let cw = containerWidth - 10;
  let ch = containerHeight - 10;

  if (cw / GridWidth > ch / GridHeight)
  {
    cw = Math.round(ch / GridHeight * GridWidth);
  }
  else
  {
    ch = Math.round(cw / GridWidth * GridHeight);
  }

  canvas.style.width = `${cw}px`;
  canvas.style.height = `${ch}px`;
  canvas.width = cw * window.devicePixelRatio;
  canvas.height = ch * window.devicePixelRatio;
}


let sbsResizeObserver = null;

function OnLoad()
{
  const gdc = document.getElementById("gridDisplayContainer");
  containerWidth = gdc.clientWidth;
  containerHeight = gdc.clientHeight;
  UpdateSizeText();
  RenderGridDisplay();

  canvas = document.getElementById("gridDisplay");
  canvas.addEventListener(
    "mousemove",
    (e) =>
    {
      const rect = e.target.getBoundingClientRect();
      mousePxX = (e.clientX - rect.left) * canvas.width / (rect.right - rect.left);
      mousePxY = (e.clientY - rect.top) * canvas.height / (rect.bottom - rect.top);
      RenderGridDisplay();
    });

  canvas.addEventListener(
    "mouseleave",
    () =>
    {
      mousePxX = -100;
      mousePxY = -100;
      RenderGridDisplay();
    });

  sbsResizeObserver = new ResizeObserver(
    (entries) =>
    {
      containerWidth = gdc.clientWidth;
      containerHeight = gdc.clientHeight;

      UpdateSizeText();
      RenderGridDisplay();
    });

  sbsResizeObserver.observe(gdc);

  let lengthSlider = document.getElementById("lengthSlider");
  lengthSlider.value = Math.round(Math.log2(FFTLength));
  lengthSlider.addEventListener(
    "input",
    () =>
    {
      FFTLength = 1 << parseInt(lengthSlider.value);
      UpdateSizeText();
      RenderGridDisplay();
    });

  let stepSlider = document.getElementById("stepSlider");
  stepSlider.value = Radix2Step * 2 / radix;
  stepSlider.addEventListener(
    "input",
    () =>
    {
      Radix2Step = parseInt(stepSlider.value * radix / 2);
      UpdateSizeText();
      RenderGridDisplay();
    });

  let flipButton = document.getElementById("bitFlipToggle");
  flipButton.addEventListener(
    "click",
    () =>
    {
      if (bitFlipMode == BitFlipMode.MiddleOut)
        { bitFlipMode = BitFlipMode.Progressive; }
      else if (bitFlipMode == BitFlipMode.Progressive)
        { bitFlipMode = BitFlipMode.Standard; }
      else if (bitFlipMode == BitFlipMode.Standard)
        { bitFlipMode = BitFlipMode.Reversed; }
      else
        { bitFlipMode = BitFlipMode.MiddleOut; }

      UpdateSizeText();
      RenderGridDisplay();
    });

  let simdButton = document.getElementById("simdToggle");
  simdButton.addEventListener(
    "click",
    () =>
    {
      if (SIMDSize == 8)
        { SIMDSize = 4; }
      else if (SIMDSize == 4)
        { SIMDSize = 2; }
      else if (SIMDSize == 2)
        { SIMDSize = 1; }
      else
        { SIMDSize = 8; }
      UpdateSizeText();
      RenderGridDisplay();
    });
  let indexButton = document.getElementById("showIndicesToggle");
  indexButton.addEventListener(
    "click",
    () =>
    {
      if (display == DisplayType.AngleIndices)
        { display = DisplayType.ElementIndices; }
      else if (display == DisplayType.ElementIndices)
        { display = DisplayType.None; }
      else
        { display = DisplayType.AngleIndices; }

      UpdateSizeText();
      RenderGridDisplay();
    });

  let radixButton = document.getElementById("radixToggle");
  radixButton.addEventListener(
    "click",
    () =>
    {
      if (radix == 2)
        { radix = 4; }
      else
        { radix = 2; }

      UpdateSizeText();
      RenderGridDisplay();
    });
}


window.onload = OnLoad;