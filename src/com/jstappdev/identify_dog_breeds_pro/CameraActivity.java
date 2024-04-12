package com.jstappdev.identify_dog_breeds_pro;

/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 * Modifications copyright (C) 2018 Josef Steppan
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import static com.jstappdev.identify_dog_breeds_pro.ListAdapter.getBitmapFromAsset;

import android.Manifest;
import android.app.ActionBar;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.TransitionDrawable;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.text.Html;
import android.text.SpannableString;
import android.text.method.LinkMovementMethod;
import android.text.util.Linkify;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.fragment.app.FragmentActivity;

import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.jstappdev.identify_dog_breeds_pro.env.ImageUtils;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;


public abstract class CameraActivity extends FragmentActivity
        implements OnImageAvailableListener, Camera.PreviewCallback {

    static final int PICK_IMAGE = 100;
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE_READ = Manifest.permission.READ_EXTERNAL_STORAGE;
    private static final String PERMISSION_STORAGE_WRITE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    static private final int[] CHART_COLORS = {Color.rgb(114, 147, 203),
            Color.rgb(225, 151, 76), Color.rgb(132, 186, 91), Color.TRANSPARENT};

    public static List<String> supportedLanguageNames;
    public static List<String> supportedLanguageCodes;

    public static String cameraId;
    private static int cameraPermissionRequests = 0;
    protected ArrayList<String> currentRecognitions;
    protected ArrayList<String> currentRecognitionsText;
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    protected ClassifierActivity.InferenceTask inferenceTask;
    TextView resultsView;
    PieChart mChart;
    AtomicBoolean snapShot = new AtomicBoolean(false);
    boolean continuousInference = false;
    boolean imageSet = false;
    ImageButton cameraButton, shareButton, closeButton, saveButton;
    ToggleButton continuousInferenceButton;
    CheckBox highAccuracyCheckbox;
    ImageView imageView;
    ProgressBar progressBar;
    private Handler handler;
    private HandlerThread handlerThread;
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private boolean useCamera2API;
    private String fileUrl;
    private boolean alreadyAdded = false;
    boolean highAccuracy = false;

    public static String preferredLanguageCode;

    abstract void handleSendImage(Intent intent);

    abstract void handleSendImageUri(Uri uri);

    // Registers a photo picker activity launcher in single-select mode.
    ActivityResultLauncher<PickVisualMediaRequest> pickMedia =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri != null) handleSendImageUri(uri);
            });

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        highAccuracy = PreferenceManager.getDefaultSharedPreferences(getBaseContext()).getBoolean("high_accuracy", false);
        preferredLanguageCode = PreferenceManager.getDefaultSharedPreferences(getBaseContext()).getString("lang", Locale.getDefault().getLanguage());
        supportedLanguageNames = Arrays.asList(getResources().getStringArray(R.array.array_languages));
        supportedLanguageCodes = Arrays.asList(getResources().getStringArray(R.array.array_language_codes));

        setLocale();

        setContentView(R.layout.activity_camera);

        setupButtons();
        setupPieChart();
        initClassifier();

        // Get intent, action and MIME type
        final Intent intent = getIntent();
        final String action = intent.getAction();
        final String type = intent.getType();

        // Handle single image being sent from other application
        if (Intent.ACTION_SEND.equals(action) && type != null) {
            if (type.startsWith("image/")) {
                if (inferenceTask != null)
                    inferenceTask.cancel(true);

                handleSendImage(intent);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, @NonNull final String[] permissions, @NonNull final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            switch (permissions[0]) {
                case PERMISSION_CAMERA:
                    setFragment();
                    break;
                case PERMISSION_STORAGE_READ:
                    pickImage();
                    break;
                case PERMISSION_STORAGE_WRITE:
                    // user might have clicked on save or share button
                    //shareButton.callOnClick();
                    break;
            }
        }
    }

    private void setupButtons() {
        imageView = findViewById(R.id.imageView);
        resultsView = findViewById(R.id.results);
        mChart = findViewById(R.id.chart);
        progressBar = findViewById(R.id.progressBar);

        continuousInferenceButton = findViewById(R.id.continuousInferenceButton);
        cameraButton = findViewById(R.id.cameraButton);
        shareButton = findViewById(R.id.shareButton);
        closeButton = findViewById(R.id.closeButton);
        saveButton = findViewById(R.id.saveButton);

        cameraButton.setEnabled(false);

        setButtonsVisibility(View.GONE);

        cameraButton.setOnClickListener(v -> {
            if (!hasPermission(PERMISSION_CAMERA)) {
                requestPermission(PERMISSION_CAMERA);
                return;
            }

            final View pnlFlash = findViewById(R.id.pnlFlash);

            cameraButton.setEnabled(false);
            snapShot.set(true);
            imageSet = false;
            updateResults(null);

            imageView.setVisibility(View.GONE);
            continuousInferenceButton.setChecked(false);

            // show flash animation
            pnlFlash.setVisibility(View.VISIBLE);
            AlphaAnimation fade = new AlphaAnimation(1, 0);
            fade.setDuration(500);
            fade.setAnimationListener(new Animation.AnimationListener() {
                @Override
                public void onAnimationStart(Animation animation) {
                }

                @Override
                public void onAnimationEnd(Animation anim) {
                    pnlFlash.setVisibility(View.GONE);
                }

                @Override
                public void onAnimationRepeat(Animation animation) {

                }
            });

            pnlFlash.startAnimation(fade);
        });

        continuousInferenceButton.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (!hasPermission(PERMISSION_CAMERA)) requestPermission(PERMISSION_CAMERA);

            imageView.setVisibility(View.GONE);
            continuousInference = isChecked;

            if (!continuousInference)
                if (inferenceTask != null)
                    inferenceTask.cancel(true);

            if (!isChecked)
                resultsView.setEnabled(false);

            cameraButton.setEnabled(true);

            imageSet = false;

            if (handler != null)
                handler.post(() -> updateResults(null));

            readyForNextImage();
        });

        resultsView.setOnClickListener(v -> {
            if (currentRecognitions == null || continuousInference || currentRecognitions.size() == 0)
                return;

            final Intent i = new Intent(getApplicationContext(), SimpleListActivity.class);
            i.putStringArrayListExtra("recogs", currentRecognitions);
            startActivity(i);
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        final MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.mainmenu, menu);
        menu.findItem(R.id.high_accuracy).setChecked(highAccuracy);
        return true;
    }

    public void setLocale() {
        Locale locale;

        if (supportedLanguageCodes.contains(preferredLanguageCode)) {
            locale = new Locale(preferredLanguageCode);
        } else {
            locale = Locale.getDefault();
        }

        Locale.setDefault(locale);
        Configuration config = new Configuration();
        config.locale = locale;
        getBaseContext().getResources().updateConfiguration(config, getBaseContext().getResources().getDisplayMetrics());    //restart Activity
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_about:

                final SpannableString s = new SpannableString(Html.fromHtml(getString(R.string.about_message)));
                Linkify.addLinks(s, Linkify.WEB_URLS);

                final AlertDialog dialog = new AlertDialog.Builder(CameraActivity.this)
                        .setMessage(s)
                        .setCancelable(true)
                        .setPositiveButton(
                                android.R.string.ok,
                                (d, id) -> d.cancel())
                        .create();

                dialog.show();

                ((TextView) dialog.findViewById(android.R.id.message)).
                        setMovementMethod(LinkMovementMethod.getInstance());

                break;
            case R.id.pick_image:
                continuousInference = false;
                continuousInferenceButton.setChecked(false);
                // use old permissions system if API level < 30,
                // photoPicker needs google services and internet access for installation
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
                    pickImage();
                } else pickMedia.launch(new PickVisualMediaRequest.Builder()
                        .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                        .build());
                break;
            case R.id.set_language:
                final LayoutInflater inflater = getLayoutInflater();
                final View dialogView = inflater.inflate(R.layout.lang_dialog, null);

                final Spinner langSpinner = dialogView.findViewById(R.id.spinner1);
                final ArrayAdapter<String> dataAdapter = new ArrayAdapter<>(this,
                        android.R.layout.simple_spinner_item, supportedLanguageNames);

                dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                langSpinner.setAdapter(dataAdapter);
                langSpinner.setSelection(supportedLanguageCodes.indexOf(preferredLanguageCode));

                new AlertDialog.Builder(CameraActivity.this)
                        .setTitle(R.string.change_language)
                        .setCancelable(true)
                        .setView(dialogView)
                        .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int id) {
                                preferredLanguageCode = supportedLanguageCodes.get(langSpinner.getSelectedItemPosition());
                                PreferenceManager.getDefaultSharedPreferences(getBaseContext()).edit().putString("lang", preferredLanguageCode).apply();
                                recreate();
                            }
                        })
                        .setIcon(R.drawable.ic_change_language)
                        .show();

                break;
            case R.id.list_breeds:
                startActivity(new Intent(this, SimpleListActivity.class));
                break;
            case R.id.high_accuracy:
                highAccuracy = !highAccuracy;
                PreferenceManager.getDefaultSharedPreferences(getBaseContext()).edit().putBoolean("high_accuracy", highAccuracy).apply();
                item.setChecked(highAccuracy);
                break;
            case R.id.action_exit:
                finishAndRemoveTask();
                break;
            default:
                break;
        }

        return true;
    }

    private void pickImage() {
        final Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        continuousInferenceButton.setChecked(false);
        startActivityForResult(i, PICK_IMAGE);
    }

    protected int[] getRgbBytes() {
        if (imageConverter != null)
            imageConverter.run();

        return rgbBytes;
    }

    /**
     * Callback for android.hardware.Camera API
     */
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        if (isProcessingFrame) {
            return;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception ignored) {
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;

        imageConverter =
                () -> ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);

        postInferenceCallback =
                () -> {
                    camera.addCallbackBuffer(bytes);
                    isProcessingFrame = false;
                };
        processImage();
    }

    /**
     * Callback for Camera2 API
     */
    @Override
    public void onImageAvailable(final ImageReader reader) {
        //We need to wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter = () -> ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            postInferenceCallback = () -> {
                image.close();
                isProcessingFrame = false;
            };

            processImage();
        } catch (final Exception ignored) {
        }
    }

    @Override
    public synchronized void onResume() {
        super.onResume();

        if (!hasPermission(PERMISSION_CAMERA)) {

            if (cameraPermissionRequests++ < 3) {
                requestPermission(PERMISSION_CAMERA);
            } else {
                Toast.makeText(getApplicationContext(), "Camera permission required.", Toast.LENGTH_LONG).show();
            }
        } else {
            setFragment();
        }

        snapShot.set(false);

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        final ActionBar actionBar = getActionBar();
        if (actionBar != null)
            actionBar.setDisplayOptions(ActionBar.DISPLAY_SHOW_HOME
                    | ActionBar.DISPLAY_SHOW_TITLE | ActionBar.DISPLAY_SHOW_CUSTOM);

        if (!imageSet) cameraButton.setEnabled(true);
    }

    private void setupPieChart() {
        mChart.getDescription().setEnabled(false);
        mChart.setUsePercentValues(true);
        mChart.setTouchEnabled(false);

        // show center text only first time
        final SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getBaseContext());
        final boolean previouslyStarted = prefs.getBoolean("showhelp", false);
        if (!previouslyStarted) {
            SharedPreferences.Editor edit = prefs.edit();
            edit.putBoolean("showhelp", Boolean.TRUE);
            edit.apply();

            mChart.setCenterTextTypeface(Typeface.createFromAsset(getAssets(), "OpenSans-Regular.ttf"));
            mChart.setCenterTextSizePixels(23);
            mChart.setDrawCenterText(true);
        }

        mChart.setExtraOffsets(14, 0.f, 14, 0.f);
        mChart.setHoleRadius(85);
        mChart.setHoleColor(Color.TRANSPARENT);
        mChart.setHovered(true);
        mChart.setDrawMarkers(false);
        mChart.setRotationEnabled(false);
        mChart.setHighlightPerTapEnabled(false);
        mChart.getLegend().setEnabled(false);
        mChart.setAlpha(0.9f);

        // display unknown slice
        final ArrayList<PieEntry> entries = new ArrayList<>();
        // set unknown slice to transparent
        entries.add(new PieEntry(100, ""));
        final PieDataSet set = new PieDataSet(entries, "");
        set.setColor(R.color.transparent);
        set.setDrawValues(false);

        final PieData data = new PieData(set);
        mChart.setData(data);
    }

    @Override
    public synchronized void onPause() {
        snapShot.set(false);
        cameraButton.setEnabled(false);
        isProcessingFrame = false;
        progressBar.setVisibility(View.GONE);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException ignored) {
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        super.onDestroy();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    private boolean hasPermission(final String permission) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission(final String permission) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{permission}, PERMISSIONS_REQUEST);
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    protected String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = isHardwareLevelSupported(characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);

                return cameraId;
            }
        } catch (CameraAccessException ignored) {
        }

        return null;
    }

    protected void setFragment() {
        cameraId = chooseCamera();
        if (cameraId == null) {
            Toast.makeText(getApplicationContext(), "No Camera Detected", Toast.LENGTH_SHORT).show();
            finish();
        }

        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, useCamera2API ? new CameraConnectionFragment() : new LegacyCameraConnectionFragment())
                .commitAllowingStateLoss();
    }

    protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                //LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    protected void readyForNextImage() {
        // sometimes this will be uninitialized, for whatever reason
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        } else isProcessingFrame = false;

    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    protected abstract void processImage();

    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    protected abstract void initClassifier();

    void updateResults(List<Classifier.Recognition> results) {
        runOnUiThread(() -> {
            updateResultsView(results);
            updatePieChart(results);
        });
    }

    void setButtonsVisibility(final int visibility) {
        final boolean enabled = visibility == View.VISIBLE;

        shareButton.setVisibility(visibility);
        shareButton.setEnabled(enabled);
        closeButton.setVisibility(visibility);
        closeButton.setEnabled(enabled);
        saveButton.setVisibility(visibility);
        saveButton.setEnabled(enabled);
    }

    // update results on our custom textview
    void updateResultsView(List<Classifier.Recognition> results) {
        final StringBuilder sb = new StringBuilder();
        currentRecognitions = new ArrayList<String>();
        currentRecognitionsText = new ArrayList<String>();

        if (results != null) {
            resultsView.setEnabled(true);

            if (!continuousInference) {
                setButtonsVisibility(View.VISIBLE);
            }

            if (results.size() > 0) {
                for (final Classifier.Recognition recog : results) {
                    final String text = String.format(Locale.getDefault(), "%s: %d %%\n",
                            recog.getTitle(), Math.round(recog.getConfidence() * 100));
                    sb.append(text);
                    currentRecognitions.add(recog.getTitle());
                    currentRecognitionsText.add(text);
                }
            } else {
                sb.append(getString(R.string.no_detection));
            }
        } else {
            resultsView.setEnabled(false);
        }

        final String finalText = sb.toString();
        resultsView.setText(finalText);
    }

    void updatePieChart(List<Classifier.Recognition> results) {
        final ArrayList<PieEntry> entries = new ArrayList<>();
        float sum = 0;

        if (results != null)
            for (int i = 0; i < results.size(); i++) {
                sum += results.get(i).getConfidence();

                PieEntry entry = new PieEntry(results.get(i).getConfidence() * 100, results.get(i).getTitle());
                entries.add(entry);
            }

        // add unknown slice
        final float unknown = 1 - sum;
        entries.add(new PieEntry(unknown * 100, ""));

        //calculate center of slice
        final float offset = entries.get(0).getValue() * 3.6f / 2;
        // calculate the next angle
        final float end = 270f - (entries.get(0).getValue() * 3.6f - offset);

        final PieDataSet set = new PieDataSet(entries, "");

        if (entries.size() > 2)
            set.setSliceSpace(3f);

        // set slice colors
        final ArrayList<Integer> sliceColors = new ArrayList<>();

        for (int c : CHART_COLORS)
            sliceColors.add(c);

        if (entries.size() > 0)
            sliceColors.set(entries.size() - 1, R.color.transparent);

        set.setColors(sliceColors);
        set.setDrawValues(false);

        final PieData data = new PieData(set);
        mChart.setData(data);

        //rotate to center of first slice
        mChart.setRotationAngle(end);
        mChart.setEntryLabelTextSize(16);
        mChart.invalidate();
    }

    protected void setImage(Bitmap image) {
        final int transitionTime = 1000;
        imageSet = true;
        alreadyAdded = false;

        cameraButton.setEnabled(false);
        imageView.setImageBitmap(image);
        imageView.setVisibility(View.VISIBLE);

        final TransitionDrawable transition = (TransitionDrawable) imageView.getBackground();
        transition.startTransition(transitionTime);

        setupShareButton();

        // fade out image on click
        final AlphaAnimation fade = new AlphaAnimation(1, 0);
        fade.setDuration(transitionTime);

        fade.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {
                if (inferenceTask != null)
                    inferenceTask.cancel(true);

                imageView.setClickable(false);
                runInBackground(() -> updateResults(null));
                transition.reverseTransition(transitionTime);
                imageView.setVisibility(View.GONE);
                setButtonsVisibility(View.GONE);
            }

            @Override
            public void onAnimationEnd(Animation anim) {
                progressBar.setVisibility(View.GONE);
                imageSet = false;
                snapShot.set(false);
                cameraButton.setEnabled(true);
                readyForNextImage();
            }

            @Override
            public void onAnimationRepeat(Animation animation) {

            }
        });

        imageView.setVisibility(View.VISIBLE);
        closeButton.setOnClickListener(v -> imageView.startAnimation(fade));

        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                saveImage();
            }
        });

    }

    public Bitmap takeScreenshot() {
        String[] fileNames = getResources().getStringArray(R.array.file_names);
        List<String> dogbreeds = Arrays.asList(getResources().getStringArray(R.array.breeds_array));
        String appname = getString(R.string.app_name);

        Bitmap icon = null;
        Drawable drawable = getDrawable(R.drawable.ic_launcher);
        if (drawable instanceof BitmapDrawable) {
            BitmapDrawable bitmapDrawable = (BitmapDrawable) drawable;
            if (bitmapDrawable.getBitmap() != null) {
                icon = bitmapDrawable.getBitmap();
            }
        } else if (drawable.getIntrinsicWidth() <= 0 || drawable.getIntrinsicHeight() <= 0) {
            icon = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888); // Single color bitmap will be created of 1x1 pixel
        } else {
            icon = Bitmap.createBitmap(drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight(), Bitmap.Config.ARGB_8888);
        }

        final View img = imageView;
        img.setDrawingCacheEnabled(true);
        img.buildDrawingCache();
        final Bitmap b = img.getDrawingCache();
        Bitmap result = Bitmap.createBitmap(b.getWidth(), b.getHeight(), b.getConfig());
        Canvas canvas = new Canvas(result);
        canvas.drawBitmap(b, 0f, 0, null);

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.DITHER_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize((int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 18, getResources().getDisplayMetrics()));
        textPaint.setTextAlign(Paint.Align.LEFT);
        Paint.FontMetrics metric = textPaint.getFontMetrics();
        int textHeight = (int) Math.ceil(metric.descent - metric.ascent);
        int y = (int) (textHeight - metric.descent);
        canvas.drawBitmap(icon, 64, b.getHeight() - icon.getHeight() - 64, null);
        canvas.drawText(appname, 84 + icon.getWidth(), b.getHeight() - icon.getHeight() - 64 + 2 * y, textPaint);

        for (int i = 0; i < currentRecognitions.size(); i++) {
            Bitmap image = getBitmapFromAsset(getApplicationContext(), fileNames[dogbreeds.indexOf(currentRecognitions.get(i))]);
            Bitmap bitmap = Bitmap.createScaledBitmap(image, 128, 128, true);
            canvas.drawBitmap(bitmap, 16, 16 + i * 128 + i * 16, null);
            canvas.drawText(currentRecognitionsText.get(i), 164, 144 + i * 128 - y + i * 16, textPaint);
        }

        return result;
    }

    private void saveImage() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R && !hasPermission(PERMISSION_STORAGE_WRITE)) {
            requestPermission(PERMISSION_STORAGE_WRITE);
            return;
        }

        if (!alreadyAdded) {
            final String fileName = getString(R.string.app_name) + " " + System.currentTimeMillis() / 1000;
            fileUrl = MediaStore.Images.Media.insertImage(getContentResolver(), takeScreenshot(), fileName, currentRecognitions.toString());
            alreadyAdded = true;
        }

        saveButton.setVisibility(View.GONE);
        saveButton.setEnabled(false);
    }

    protected void setupShareButton() {
        shareButton.setOnClickListener(v -> {
            saveImage();

            final Uri contentUri = Uri.parse(fileUrl);

            if (contentUri != null) {
                Intent shareIntent = new Intent();
                shareIntent.setAction(Intent.ACTION_SEND);
                shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION); // temp permission for receiving app to read this file
                shareIntent.setDataAndType(contentUri, getContentResolver().getType(contentUri));
                shareIntent.putExtra(Intent.EXTRA_STREAM, contentUri);
                startActivity(Intent.createChooser(shareIntent, getString(R.string.share_with)));
            }
        });
    }
}