package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;
// import com.fasterxml.jackson.databind.SerializationFeature;
import com.github.gumtreediff.actions.EditScript;
import com.github.gumtreediff.actions.EditScriptGenerator;
import com.github.gumtreediff.actions.InsertDeleteChawatheScriptGenerator;
import com.github.gumtreediff.client.Run;
import com.github.gumtreediff.gen.TreeGenerators;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.Tree;
import com.github.gumtreediff.tree.TreeContext;

import java.io.*;

public class GumTreeDiffParserProxy {
    public static void main(String[] args) throws IOException {
        ProcessFiles(args[0], args[1], "new_code_tree.json", "edit_script_info.json");
    }

    public static void ProcessFiles(String newFile, String oldFile, String astOutput, String actionOutput) throws IOException {
        Run.initGenerators();
        Run.initClients();
        TreeGenerators defaultTreeGen = TreeGenerators.getInstance();
        Matcher defaultMatcher = Matchers.getInstance().getMatcher();
        EditScriptGenerator editScriptGenerator = new InsertDeleteChawatheScriptGenerator();

        TreeContext old_tree_context = defaultTreeGen.getTree(oldFile);
        TreeContext new_tree_context = defaultTreeGen.getTree(newFile);
        MappingStore mappings = defaultMatcher.match(old_tree_context.getRoot(), new_tree_context.getRoot());
        EditScript actions = editScriptGenerator.computeActions(mappings);

//        ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        ObjectMapper objectMapper = new ObjectMapper();


        var data = make_serializable_tree(new_tree_context.getRoot());
        objectMapper.writeValue(new File(astOutput), data);

        var serializable_edit_script = make_serializable_edit_script(actions);
        objectMapper.writeValue(new File(actionOutput), serializable_edit_script);

    }

    private static TreeData make_serializable_edit_script(EditScript actions) {
        TreeData[] actionsMapped = new TreeData[actions.size()];
        var i = 0;
        for(var act : actions){
            var actNode = act.getNode();
            var actNodeData = make_serializable_tree(actNode);
            actionsMapped[i] = new TreeData(act.getName(), actNodeData);
            i++;
        }
        return new TreeData(actionsMapped);
    }

    private static TreeData make_serializable_tree(Tree t) {
        var children = t.getChildren();
        if (children.isEmpty()) {
            return new TreeData(t, new TreeData[0]);
        }

        TreeData[] arr = new TreeData[children.size()];
        var i = 0;
        for (var child : children){
            var temp = make_serializable_tree(child);
            arr[i] = temp;
            i++;
        }
        return new TreeData(t, arr);
    }
}