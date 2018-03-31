package com.mirismr;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;

public class Tree {

    public static DefaultMutableTreeNode buildTree(JSONObject padre) {
        DefaultMutableTreeNode root = new DefaultMutableTreeNode(padre.get("name"));
        JSONArray children = (JSONArray) padre.get("children");
        if (children != null) {
            for (int i = 0; i < children.size(); i++) {
                DefaultMutableTreeNode child = buildTree((JSONObject) children.get(i));
                root.add(child);
            }
        }
        return root;
    }

    public static TreePath find(DefaultMutableTreeNode root, String s) {
        Enumeration<DefaultMutableTreeNode> e = root.depthFirstEnumeration();
        while (e.hasMoreElements()) {
            DefaultMutableTreeNode node = e.nextElement();
            if (node.toString().equalsIgnoreCase(s)) {
                return new TreePath(node.getPath());
            }
        }
        return null;
    }

    public static void main(String[] args) {
        Object obj = null;
        try {
            obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00007846.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));
            //obj = new JSONParser().parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\json_n00015388.json"));

        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }

        JSONObject jo = (JSONObject) obj;
        DefaultMutableTreeNode raiz = buildTree(jo);
        TreePath path = find(raiz, "n10557404");
        System.out.println(path);
    }
}
