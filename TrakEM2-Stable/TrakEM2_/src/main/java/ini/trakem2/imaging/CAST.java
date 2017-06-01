package ini.trakem2.imaging;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.PointRoi;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.process.ImageProcessor;
import ij.ImageStack;
import trainableSegmentation.WekaSegmentation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import ini.trakem2.display.AreaContainer;
import ini.trakem2.display.AreaWrapper;
import ini.trakem2.display.AreaList;
import ini.trakem2.display.Display;
import ini.trakem2.display.Layer;
import ini.trakem2.display.Patch;
import ini.trakem2.utils.Bureaucrat;
import ini.trakem2.utils.IJError;
import ini.trakem2.utils.M;
import ini.trakem2.utils.OptionPanel;
import ini.trakem2.utils.Utils;
import ini.trakem2.utils.Worker;

import java.awt.Choice;
import java.awt.Checkbox;
import java.awt.Color;
import java.awt.Component;
import java.awt.Rectangle;
import java.awt.geom.AffineTransform;
import java.awt.geom.Area;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import levelsets.algorithm.Coordinate;

public class CAST {
	
	static public class Witchdoctor {
		
		private File model;
		private String parentfolder = new File(CAST.class.getProtectionDomain().getCodeSource().getLocation().getPath()).getAbsoluteFile().getParent();
		private String traindata;
		private String modelFile;
		private String[] labels = new String[2];
		private WekaSegmentation segment;
		private MultilayerPerceptron nnet;
		private ImagePlus classifiedImage;
		
		Witchdoctor(ImagePlus imp, String fileName){
			// setting up the Witchdoctor.
			traindata = parentfolder+File.separator+"trainingData"+File.separator+fileName+".arff";
			modelFile = parentfolder+File.separator+"models"+File.separator+fileName+".model";
			model = new File(modelFile);
			segment = new WekaSegmentation(imp);
			nnet = new weka.classifiers.functions.MultilayerPerceptron();
			classifiedImage = getClassification();
			//this will show you what expand area sees
			//classifiedImage.show();
		}
		
		/* This function is called on initalization
		 * The function analyzes the image that was passed
		 * Returns a binary ImagePlus object
		 * The algorithm marks what it thinks is area as black.
		 */
		public ImagePlus getClassification(){
			segment.setClassifier(nnet);
			try{
				BufferedReader reader = new BufferedReader(new FileReader(traindata));
				String line;
				while((line = reader.readLine()) != null){
					String[] splited = line.split(" ");
					if(Arrays.asList(splited).contains("@attribute") && Arrays.asList(splited).contains("class")){
						String classString = line.substring(line.indexOf('{')+1,line.indexOf('}'));
						String newClassString = classString.replace("\'","");
						String[] classes = newClassString.split(",");
						labels[0] = classes[0];
						labels[1] = classes[1];
						break;
					}
				}
				reader.close();
				segment.setClassLabels(labels);
			} catch (Exception e){
				Utils.log2("class name: "+labels[0]);
				Utils.log2("class name: "+labels[1]);
			}
			
			//check to see if a model exists.
			if(model.exists() && !model.isDirectory()){
				segment.loadClassifier(modelFile);
			}
			// if there isn't a model build one!
			else{
				segment.loadTrainingData(traindata);
				segment.trainClassifier();
				segment.saveClassifier(modelFile);
			}
			// classify the image and return
			segment.applyClassifier(false);
			ImagePlus result = segment.getClassifiedImage();
			return result;
		}
		
		/*This function takes in the x,y coordinates of the previously Analyzed image.
		 *Returns true if the point is a boundary...(not part of the inner area of the structure).
		 */
		public boolean isBoundary(int x, int y){
			// the point is not black, so it must be a boundary... or something else.
			if(classifiedImage.getPixel(x, y)[0]!=0){
				return true;
			}
			else{
				return false;
			}
		}
		
	}
	
	static public class ComparablePoint extends Point implements Comparable<ComparablePoint> {
		ComparablePoint(int x, int y) {
			super(x, y);
		}
		public int compareTo(ComparablePoint p) {
			int p_num = p.x * 100000 + p.y;
			int our_num = x * 100000 + y;
			return our_num - p_num;
		}
	}
	
	static public class VoodooParams {
		// Permissible Deviation
		public double deviation = 0.5;
		// Permissible Deviation
		public int boundaryDistance = 1;
		// Name of the desired training file
		public String fileName = getNames()[0];
		//This function grabs the names of the files in the "trainingData" directory
		public String[] getNames(){
			String parentfolder = new File(CAST.class.getProtectionDomain().getCodeSource().getLocation().getPath()).getAbsoluteFile().getParent();
			String path = parentfolder+File.separator+"trainingData";
			File f = new File(path);
			File[] subFiles = f.listFiles();
			String[] fnames = new String[subFiles.length];
			for(int i = 0; i < subFiles.length; i++){
				if(subFiles[i].isFile()){
					fnames[i] = subFiles[i].getName();
				}
			}
			return fnames;
		}
		
		public String[] names = getNames();
		/** In pixels, of the the underlying image to copy around the mouse. May be enlarged by shift+scrollwheel with PENCIL tool on a selected Displayable. */
		public int width = 100, height = 100;
		
		public OptionPanel asOptionPanel() {
			OptionPanel op = new OptionPanel();
			op.addMessage("Computer Assisted Segmentation: ");
			op.addNumericField("Boundary Padding:", boundaryDistance, new OptionPanel.IntSetter(this, "boundaryDistance"));
			op.addChoice("Training Set(s): ", names, 0, new OptionPanel.ChoiceStringSetter(this, "fileName"));
			return op;
		}
		
		public boolean setup() {
			GenericDialog gd = new GenericDialog("Automatic Segmentation Options");
			gd.addMessage("Automatic Segmentation Tool:");
			gd.addNumericField("Boundary Padding", boundaryDistance, 0);
			gd.addChoice("Training Set(s): ", names, names[0]);
			gd.showDialog();
			if (gd.wasCanceled()) return false;
			deviation = (double) gd.getNextNumber();
			boundaryDistance = (int)gd.getNextNumber();
			fileName = (String)gd.getNextChoice();
			return true;
		}
		
		public void resizeArea(final int sign, final double magnification) {
			double inc = (int)( (10 * sign) / magnification);
			this.width += inc;
			this.height += inc;
			Utils.log2("w,h: " + width + ", " + height);
		}
		
		// Return bounds relative to the given mouse position
		public Rectangle getBounds(final int x_p, final int y_p) {
			return new Rectangle(x_p - width/2, y_p - height/2, width, height);
		}
	}
	
	static public final VoodooParams vp = new VoodooParams();
	
	static class EndTask<T,D> {
		T target;
		Runnable after;
		EndTask(T target, Runnable after) {
			this.target = target;
			this.after = after;
		}
		void run() {
		}
	}
	
	static public int neighbor_average(ImagePlus imp, int x, int y){
		int center_average = 0;
		for(int i = -10; i <= 10; i++)
			for(int j = -10; j <= 10; j++)
				center_average += imp.getPixel(x + i, y + j)[0];
		center_average /= 21 * 21;
		return center_average;
	}
	
	static public double neighbor_shift_nn(Witchdoctor fred, int x, int y){
		double max_neighbor = 0;
		double min_neighbor = 255;
		int dist = vp.boundaryDistance;
		for(int i = -dist; i <= dist; i++){
			for(int j = -dist; j <= dist; j++) {
				int color = fred.isBoundary(x+i, y+j)? 255:0;
				if(max_neighbor < color) max_neighbor = color;
				if(min_neighbor > color) min_neighbor = color;
			}
		}
		return (max_neighbor - min_neighbor) / ((max_neighbor + min_neighbor)/2.0 + 1);
	}
	
	static public int getcolor(ImagePlus imp, int x, int y){
		return imp.getPixel(x, y)[0];
	}
	
	static public Bureaucrat Voodoo(final AreaWrapper aw, final Layer layer, final Rectangle srcRect, final int x_p_w, final int y_p_w, final Runnable post_task) {
		return Voodoo(aw, layer, srcRect, x_p_w, y_p_w, Arrays.asList(new Runnable[]{post_task}));
	}
	
	static public Bureaucrat Voodoo(final AreaWrapper aw, final Layer layer, final Rectangle srcRect, final int x_p_w, final int y_p_w, final List<Runnable> post_tasks) {
		// Capture pointers before they are set to null
		final AreaContainer ac = (AreaContainer)aw.getSource();
		final AffineTransform source_aff = aw.getSource().getAffineTransform();
		final Rectangle box = new Rectangle(x_p_w - CAST.vp.width/2, y_p_w - CAST.vp.height/2, CAST.vp.width, CAST.vp.height);
		//sets up process for a new thread
		Bureaucrat burro = Bureaucrat.create(new Worker.Task("Performing Voodoo") { public void exec() {
			// Capture image as large as the vp width,height centered on x_p_w,y_p_w
			ImagePlus imp = (ImagePlus) layer.grab(box, 1.0, Patch.class, (int) aw.getSource().getAlpha(), Layer.IMAGEPLUS, ImagePlus.GRAY8);
			PointRoi roi = new PointRoi(box.width/2, box.height/2);
			imp.setRoi(roi);
			Utils.log2("imp: " + imp);
			Utils.log2("roi: " + imp.getRoi() + "    " + Utils.toString(new int[]{x_p_w - srcRect.x, y_p_w - srcRect.y}));
			final Area area = new Area();
			final Rectangle r = new Rectangle(0, 0, 1, 1);
			// Center of box:  box.width/2, box.height/2, and it gets translated
			ComparablePoint center = new ComparablePoint(box.width/2, box.height/2);
			int centerval = imp.getPixel(center.x, center.y)[0];
			TreeSet<ComparablePoint> frontier = new TreeSet<ComparablePoint>();
			frontier.add(center);
			TreeSet<ComparablePoint> inside = new TreeSet<ComparablePoint>();
			int center_average = neighbor_average(imp, center.x, center.y);
			Witchdoctor fred = new Witchdoctor(imp, vp.fileName.substring(0, vp.fileName.indexOf(".")));
			final int margin = 11;
			
			while(frontier.size() > 0){
				ComparablePoint current = frontier.pollFirst();
				boolean in_boundary = false;
				in_boundary = neighbor_shift_nn(fred, current.x, current.y) > vp.deviation;
				if(!in_boundary){
					r.x = current.x;
					r.y = current.y;
					area.add(new Area(r));
					inside.add(current);
					
					final ComparablePoint neighbor1 = new ComparablePoint(current.x+1, current.y);
					if(!inside.contains(neighbor1) && neighbor1.x < box.width - margin)
						frontier.add(neighbor1);
					final ComparablePoint neighbor2 = new ComparablePoint(current.x-1, current.y);
					if(!inside.contains(neighbor2) && neighbor2.x >= margin)
						frontier.add(neighbor2);
					final ComparablePoint neighbor3 = new ComparablePoint(current.x, current.y+1);
					if(!inside.contains(neighbor3) && neighbor3.y < box.height - margin)
						frontier.add(neighbor3);
					final ComparablePoint neighbor4 = new ComparablePoint(current.x, current.y-1);
					if(!inside.contains(neighbor4) && neighbor4.y >= margin)
						frontier.add(neighbor4);
				}
				if(inside.size() % 1000 == 0){
					Utils.log2("Inside size:  " + inside.size());
				}
			}
			final AffineTransform aff = new AffineTransform(1, 0, 0, 1, box.x, box.y);
			try {
				aff.preConcatenate(source_aff.createInverse());
			} catch (NoninvertibleTransformException nite) {
				IJError.print(nite);
				return;
			}
			aw.getArea().add(area.createTransformedArea(aff));
			ac.calculateBoundingBox(layer);
			Display.repaint(layer);
		}}, layer.getProject());
		//End of buro
		if (null != post_tasks) for (Runnable task : post_tasks) burro.addPostTask(task);
		//this starts the thread
		burro.goHaveBreakfast();
		return burro;
	}
}
