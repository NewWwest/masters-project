package org.example;

import com.github.gumtreediff.tree.Tree;

public class TreeData {
    public Integer StartByte;
    public Integer EndByte;
    public String Label;
    public TreeData[] Children;

    public TreeData(String title, TreeData child){
        StartByte = child.StartByte;
        EndByte = child.EndByte;
        Label = title;
        Children = new TreeData[]{child};
    }

    public TreeData(TreeData[] children){
        StartByte = -1;
        EndByte = -1;
        Label = "root";
        Children = children;
    }

    public TreeData(Tree t, TreeData[] children){
        StartByte = t.getPos();
        EndByte = t.getEndPos();
        Label = t.toString();
        Children = children;
    }
}
